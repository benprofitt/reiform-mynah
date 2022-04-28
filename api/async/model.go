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

// AsyncProvider interface for launching new background processes
type AsyncProvider interface {
	// StartAsyncTask start processing an async task
	StartAsyncTask(*model.MynahUser, AsyncTaskHandler) model.MynahUuid
	// GetAsyncTaskStatus gets the status of a task
	GetAsyncTaskStatus(*model.MynahUser, model.MynahUuid) (MynahAsyncTaskStatus, error)
	// Close close the async task provider
	Close()
}
