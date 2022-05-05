// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/model"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
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
func (p *localImplProvider) ICProcessJob(user *model.MynahUser, datasetId model.MynahUuid, dataset *model.MynahICDatasetVersion, tasks []model.MynahICProcessTaskType) error {
	//initialize the function
	fn, err := p.pythonProvider.InitFunction(p.mynahSettings.PythonSettings.ModuleName, "start_ic_processing_job")
	if err != nil {
		return err
	}

	//create the request
	req, err := p.NewICProcessJobRequest(user, datasetId, dataset, tasks)
	if err != nil {
		return err
	}

	//call the function
	res := fn.Call(user, req)

	var jobResponse ICProcessJobResponse

	//parse the response
	resErr := res.GetResponse(&jobResponse)
	if resErr != nil {
		return resErr
	}

	//apply the changes
	return jobResponse.apply(dataset)
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
