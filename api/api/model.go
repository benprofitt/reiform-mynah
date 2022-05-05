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

// ICProcessJobRequest request type for starting a process job
type ICProcessJobRequest struct {
	//The tasks to perform
	Tasks []model.MynahICProcessTaskType `json:"tasks"`
	//the dataset id
	DatasetUuid model.MynahUuid `json:"dataset_uuid"`
}

// ICProcessJobResponse response type for start process job
type ICProcessJobResponse struct {
	//the id of the task
	TaskUuid model.MynahUuid `json:"task_uuid"`
}

// TaskStatusResponse response type for querying the status of a task
type TaskStatusResponse struct {
	//the status of the task
	TaskStatus async.MynahAsyncTaskStatus `json:"task_status"`
}
