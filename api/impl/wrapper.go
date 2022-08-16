// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

import (
	"encoding/json"
	"fmt"
	"reiform.com/mynah/db"
	"reiform.com/mynah/model"
	"reiform.com/mynah/mynahExec"
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

// GetMynahImplVersion request the current version of python tools
func (p *localImplProvider) GetMynahImplVersion() (*VersionResponse, error) {
	var user model.MynahUser
	var versionRes VersionResponse
	return &versionRes, p.mynahExecutor.Call(&user, "get_impl_version", nil).GetAs(&versionRes)
}

// ICProcessJob start a ic process job
func (p *localImplProvider) ICProcessJob(user *model.MynahUser,
	dataset *model.MynahICDataset,
	tasks []model.MynahICProcessTaskType) error {
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
		if err = jobResponse.applyChanges(newVersion, report, user, p.storageProvider, p.dbProvider); err != nil {
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
	return err
}

// ImageMetadata get image metadata
func (p *localImplProvider) ImageMetadata(user *model.MynahUser, path string, file *model.MynahFile, version *model.MynahFileVersion) error {
	var response ImageMetadataResponse
	err := p.mynahExecutor.Call(user, "get_image_metadata", p.NewImageMetadataRequest(path)).GetAs(&response)

	if err != nil {
		return fmt.Errorf("failed to execute get_image_metadata: %s", err)
	}

	//modify the file
	return response.apply(file, version)
}
