// Copyright (c) 2022 by Reiform. All Rights Reserved.

package async

import (
	"reiform.com/mynah/model"
)

// MynahAsyncTaskStatus describes the status of a task
type MynahAsyncTaskStatus string

const (
	StatusPending   MynahAsyncTaskStatus = "pending"
	StatusRunning   MynahAsyncTaskStatus = "running"
	StatusCompleted MynahAsyncTaskStatus = "completed"
	StatusFailed    MynahAsyncTaskStatus = "failed"
)

// AsyncTaskHandler Handler to invoke asynchronously
type AsyncTaskHandler func(model.MynahUuid) ([]byte, error)

// AsyncTaskData defines data about a running async task
type AsyncTaskData struct {
	//when the task started
	Started int64 `json:"started"`
	//the id of the task
	TaskId model.MynahUuid `json:"task_id"`
	//the status of the task
	TaskStatus MynahAsyncTaskStatus `json:"task_status"`
}

// AsyncProvider interface for launching new background processes
type AsyncProvider interface {
	// StartAsyncTask start processing an async task
	StartAsyncTask(*model.MynahUser, AsyncTaskHandler) model.MynahUuid
	// GetAsyncTaskStatus gets the status of a task
	GetAsyncTaskStatus(*model.MynahUser, model.MynahUuid) (*AsyncTaskData, error)
	// ListAsyncTasks lists the async tasks owned by a user
	ListAsyncTasks(*model.MynahUser) (res []*AsyncTaskData)
	// Close close the async task provider
	Close()
}
