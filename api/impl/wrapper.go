// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

import (
	"reiform.com/mynah/model"
	"reiform.com/mynah/python"
)

//request the current version of python tools
func GetMynahImplVersion(pythonProvider python.PythonProvider) (*VersionResponse, error) {
	//initialize the function
	fn, err := pythonProvider.InitFunction("mynah", "get_impl_version")
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

//start a diagnosis job
func DiagnosisJob(pythonProvider python.PythonProvider,
	user *model.MynahUser,
	request *DiagnosisJobRequest) (*DiagnosisJobResponse, error) {

	//initialize the function
	fn, err := pythonProvider.InitFunction("mynah", "start_diagnosis_job")
	if err != nil {
		return nil, err
	}

	//call the function
	res := fn.Call(user, request)

	var jobResponse DiagnosisJobResponse

	//parse the response
	resErr := res.GetResponse(&jobResponse)
	if resErr != nil {
		return nil, resErr
	}
	return &jobResponse, nil
}
