// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"reiform.com/mynah/model"
	"reiform.com/mynah/python"
)

//local python impl provider
type localImplProvider struct {
	//the python provider
	pythonProvider python.PythonProvider
	//the module name
	moduleName string
}

// GetMynahImplVersion request the current version of python tools
func (p *localImplProvider) GetMynahImplVersion() (*VersionResponse, error) {
	//initialize the function
	fn, err := p.pythonProvider.InitFunction(p.moduleName, "get_impl_version")
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

// ICDiagnoseCleanJob start a diagnosis job
func (p *localImplProvider) ICDiagnoseCleanJob(user *model.MynahUser, request *ICDiagnoseCleanJobRequest) (*ICDiagnoseCleanJobResponse, error) {
	//initialize the function
	fn, err := p.pythonProvider.InitFunction(p.moduleName, "start_ic_processing_job")
	if err != nil {
		return nil, err
	}

	//call the function
	res := fn.Call(user, request)

	var jobResponse ICDiagnoseCleanJobResponse

	//parse the response
	resErr := res.GetResponse(&jobResponse)
	if resErr != nil {
		return nil, resErr
	}
	return &jobResponse, nil
}

// ImageMetadata get image metadata
func (p *localImplProvider) ImageMetadata(user *model.MynahUser, request *ImageMetadataRequest) (*ImageMetadataResponse, error) {
	//initialize the function
	fn, err := p.pythonProvider.InitFunction(p.moduleName, "get_image_metadata")
	if err != nil {
		return nil, err
	}

	//call the function
	res := fn.Call(user, request)

	var response ImageMetadataResponse

	//parse the response
	resErr := res.GetResponse(&response)
	if resErr != nil {
		return nil, resErr
	}
	return &response, nil
}
