// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"encoding/json"
	"fmt"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/tools"
)

//local python impl provider
type localImplProvider struct {
	//the python provider
	pythonProvider python.PythonProvider
	//the database provider
	dbProvider db.DBProvider
	//the storage provider
	storageProvider storage.StorageProvider
	//settings
	mynahSettings *settings.MynahSettings
}

// GetMynahImplVersion request the current version of python tools
func (p *localImplProvider) GetMynahImplVersion() (*VersionResponse, error) {
	//initialize the function
	fn, err := p.pythonProvider.InitFunction(p.mynahSettings.PythonSettings.ModuleName, "get_impl_version")
	if err != nil {
		return nil, err
	}

	var user model.MynahUser

	//call the function
	res := fn.Call(&user, nil)

	var versionResponse VersionResponse

	resErr := res.GetResponse(&versionResponse)
	if resErr != nil {
		return nil, resErr
	}
	return &versionResponse, nil
}

// ICProcessJob start a ic process job
func (p *localImplProvider) ICProcessJob(user *model.MynahUser,
	dataset *model.MynahICDataset,
	tasks []model.MynahICProcessTaskType) error {
	//initialize the function
	fn, err := p.pythonProvider.InitFunction(p.mynahSettings.PythonSettings.ModuleName, "start_ic_processing_job")
	if err != nil {
		return err
	}

	//create a new version of the dataset to operate on
	newVersion, err := tools.MakeICDatasetVersion(dataset)
	if err != nil {
		return fmt.Errorf("ic process task for dataset %s failed when creating new version: %s", dataset.Uuid, err)
	}

	//get the previous version of the dataset (if applicable) to send task metadata
	prevVersion, err := tools.GetICDatasetPrevious(dataset)
	if err != nil {
		log.Warnf("dataset %s does not have previous version to send to ic process", dataset.Uuid)
	}

	//create the request
	req, err := p.NewICProcessJobRequest(user, dataset.Uuid, newVersion, prevVersion, tasks)
	if err != nil {
		return fmt.Errorf("ic process job failed when creating request: %s", err)
	}

	//call the function
	res := fn.Call(user, req)

	var jobResponse ICProcessJobResponse

	//parse the response
	if err = res.GetResponse(&jobResponse); err != nil {
		return fmt.Errorf("ic process job failed: %s", err)
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

	//record the report by id
	dataset.Reports[dataset.LatestVersion] = binObj.Uuid
	return err
}

// ImageMetadata get image metadata
func (p *localImplProvider) ImageMetadata(user *model.MynahUser, path string, file *model.MynahFile, version *model.MynahFileVersion) error {
	//initialize the function
	fn, err := p.pythonProvider.InitFunction(p.mynahSettings.PythonSettings.ModuleName, "get_image_metadata")
	if err != nil {
		return err
	}

	//call the function
	res := fn.Call(user, p.NewImageMetadataRequest(path))

	var response ImageMetadataResponse

	//parse the response
	resErr := res.GetResponse(&response)
	if resErr != nil {
		return resErr
	}

	//modify the file
	return response.apply(file, version)
}
