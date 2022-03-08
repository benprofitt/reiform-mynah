// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"reiform.com/mynah/model"
)

//request type for an admin creating a user
type adminCreateUserRequest struct {
	//the first and last name to assign the user
	NameFirst string `json:"name_first"`
	NameLast  string `json:"name_last"`
}

//response type for an admin creating a user
type adminCreateUserResponse struct {
	//the generated jwt
	Jwt string `json:"jwt"`
	//the user itself
	User model.MynahUser `json:"user"`
}

//request type for creating a dataset
type createICDatasetRequest struct {
	//the name of the dataset
	Name string `json:"name"`
	//the files to include
	Files map[string]string `json:"files"`
}

//response type for creating a dataset
type createICDatasetResponse struct {
	//the dataset
	Dataset model.MynahICDataset `json:"dataset"`
}

//request type for creating an ic project
type createICProjectRequest struct {
	//the name for the project
	Name string `json:"name"`
	//datasets to link to this project
	Datasets []string `json:"datasets"`
}

//response type for creating an ic project
type createICProjectResponse struct {
	//the created project
	Project model.MynahICProject `json:"project"`
}

//request type for start diagnosis job
type startDiagnosisJobRequest struct {
	//the project id
	ProjectUuid string `json:"project_uuid"`
}

//response type for start diagnosis job
type startDiagnosisJobResponse struct {
	//TODO
}
