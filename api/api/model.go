// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"reiform.com/mynah/model"
)

// AdminCreateUserRequest request type for an admin creating a user
type AdminCreateUserRequest struct {
	//the first and last name to assign the user
	NameFirst string `json:"name_first"`
	NameLast  string `json:"name_last"`
}

// AdminCreateUserResponse response type for an admin creating a user
type AdminCreateUserResponse struct {
	//the generated jwt
	Jwt string `json:"jwt"`
	//the user itself
	User model.MynahUser `json:"user"`
}

// CreateICDatasetRequest request type for creating a dataset
type CreateICDatasetRequest struct {
	//the name of the dataset
	Name string `json:"name"`
	//the files to include
	Files map[string]string `json:"files"`
}

// CreateICProjectRequest request type for creating an ic project
type CreateICProjectRequest struct {
	//the name for the project
	Name string `json:"name"`
	//datasets to link to this project
	Datasets []string `json:"datasets"`
}

// CreateODProjectRequest request type for creating an od project
type CreateODProjectRequest struct {
	//the name for the project
	Name string `json:"name"`
	//datasets to link to this project
	Datasets []string `json:"datasets"`
}

// StartDiagnosisJobRequest request type for start diagnosis job
type StartDiagnosisJobRequest struct {
	//the project id
	ProjectUuid string `json:"project_uuid"`
}

// StartDiagnosisJobResponse response type for start diagnosis job
type StartDiagnosisJobResponse struct {
	//TODO
}
