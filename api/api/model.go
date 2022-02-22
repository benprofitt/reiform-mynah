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

//request type for start diagnosis job
type startDiagnosisJobRequest struct {
	//the project id
	ProjectUuid string `json:"project_uuid"`
}

//response type for start diagnosis job
type startDiagnosisJobResponse struct {
	//TODO
}
