// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

import (
	"encoding/json"
	"fmt"
	"reiform.com/mynah/containers"
	"reiform.com/mynah/db"
	"reiform.com/mynah/model"
	"reiform.com/mynah/mynahExec"
	"reiform.com/mynah/mynahSync"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/tools"
	"time"
)

//local impl provider
type localImplProvider struct {
	mynahSettings *settings.MynahSettings
	//the database provider
	dbProvider db.DBProvider
	//the storage provider
	storageProvider storage.StorageProvider
	//the executor
	mynahExecutor mynahExec.MynahExecutor
}

// NewImplProvider create a new provider
func NewImplProvider(mynahSettings *settings.MynahSettings,
	dbProvider db.DBProvider,
	storageProvider storage.StorageProvider,
	mynahExecutor mynahExec.MynahExecutor) ImplProvider {
	return &localImplProvider{
		mynahSettings:   mynahSettings,
		dbProvider:      dbProvider,
		storageProvider: storageProvider,
		mynahExecutor:   mynahExecutor,
	}
}

// GetMynahImplVersion request the current version of python tools
func (p *localImplProvider) GetMynahImplVersion() (*VersionResponse, error) {
	var user model.MynahUser
	var versionRes VersionResponse
	return &versionRes, p.mynahExecutor.Call(&user, "get_impl_version", nil).GetAs(&versionRes)
}

// ICProcessJob start a ic process job
func (p *localImplProvider) ICProcessJob(user *model.MynahUser,
	datasetId model.MynahUuid,
	tasks []model.MynahICProcessTaskType) error {
	//lock the dataset before making changes
	lock, err := mynahSync.GetSyncProvider().Lock(datasetId)
	if err != nil {
		return fmt.Errorf("failed to aquire lock for dataset before processing: %w", err)
	}

	//get the dataset
	dataset, err := p.dbProvider.GetICDataset(datasetId, user, db.NewMynahDBColumns())
	if err != nil {
		return fmt.Errorf("failed to get dataset for ic process job: %w", err)
	}

	//collect ids of files in dataset to acquire locks for
	fileIdSet := containers.NewUniqueSet[model.MynahUuid]()
	for fileId := range dataset.Versions[dataset.LatestVersion].Files {
		fileIdSet.Union(fileId)
	}

	// lock the files in the dataset
	fileLocks, err := mynahSync.GetSyncProvider().LockMany(fileIdSet.Vals())
	if err != nil {
		return fmt.Errorf("failed to acquire locks for files in dataset before processing: %s", err)
	}
	// Unlock remaining. Note: these locks _should_ be released when freezing the file versions
	defer fileLocks.CheckUnlocked()

	//create a new version of the dataset to operate on
	newVersion, err := tools.MakeICDatasetVersion(dataset)
	if err != nil {
		return fmt.Errorf("ic process task for dataset %s failed when creating new version: %s", dataset.Uuid, err)
	}

	//create the request
	req, err := p.NewICProcessJobRequest(user, newVersion, dataset, tasks)
	if err != nil {
		return fmt.Errorf("ic process job failed when creating request: %s", err)
	}

	res := p.mynahExecutor.Call(user, "start_ic_processing_job", req)

	var jobResponse ICProcessJobResponse

	if err = res.GetAs(&jobResponse); err != nil {
		return fmt.Errorf("failed to execute start_ic_processing_job: %s", err)
	}

	//create a new report based on this run
	binObj, err := p.dbProvider.CreateBinObject(user, func(binObj *model.MynahBinObject) error {
		report := model.NewICDatasetReport()

		//apply changes to dataset and report
		if err = jobResponse.applyChanges(newVersion, report, user, p.storageProvider, p.dbProvider, fileLocks); err != nil {
			return fmt.Errorf("ic process job failed when applying process changes to dataset and report: %s", err)
		}

		//store the report
		data, err := json.Marshal(report)
		if err != nil {
			return fmt.Errorf("ic process job failed when serializing report: %s", err)
		}
		//set the binary data to store
		binObj.Data = data
		return nil
	})

	if err != nil {
		return fmt.Errorf("ic process job failed when creating report: %s", err)
	}

	//record the report by id
	dataset.Reports[dataset.LatestVersion] = &model.MynahICDatasetReportMetadata{
		DataId:      binObj.Uuid,
		DateCreated: time.Now().Unix(),
		Tasks:       tasks,
	}

	//update the results in the database (will overwrite any changes made to versions col since task started)
	if err := p.dbProvider.UpdateICDataset(dataset, user, lock, db.NewMynahDBColumns(model.VersionsColName, model.ReportsColName)); err != nil {
		return fmt.Errorf("ic process task for dataset failed when updating in database: %s", err)
	}

	return nil
}

// BatchImageMetadata gets image metadata. Note: this will overwrite any metadata in the specified file version
func (p *localImplProvider) BatchImageMetadata(user *model.MynahUser, files map[model.MynahUuid]storage.MynahLocalFile) error {
	var response ImageMetadataResponse
	err := p.mynahExecutor.Call(user, "get_metadata_for_images", p.NewBatchImageMetadataRequest(files)).GetAs(&response)

	if err != nil {
		return fmt.Errorf("failed to execute get_image_metadata: %s", err)
	}

	//modify the file versions with the response
	return response.apply(files)
}
