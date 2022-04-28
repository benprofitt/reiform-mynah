// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"reiform.com/mynah/async"
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
	//the files to include (map from fileid to class name)
	Files map[model.MynahUuid]string `json:"files"`
}

// ICCleanDiagnoseJobRequest request type for starting a diagnosis/clean job
type ICCleanDiagnoseJobRequest struct {
	//whether to run the diagnosis step
	Diagnose bool `json:"diagnose"`
	//whether to run the clean step
	Clean bool `json:"clean"`
	//the dataset id
	DatasetUuid model.MynahUuid `json:"dataset_uuid"`
}

// ICCleanDiagnoseJobResponse response type for start diagnosis job
type ICCleanDiagnoseJobResponse struct {
	//the id of the task
	TaskUuid model.MynahUuid `json:"task_uuid"`
}

// TaskStatusResponse response type for querying the status of a task
type TaskStatusResponse struct {
	//the status of the task
	TaskStatus async.MynahAsyncTaskStatus `json:"task_status"`
}
