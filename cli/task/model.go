// Copyright (c) 2022 by Reiform. All Rights Reserved.

package task

import (
	"context"
	"reiform.com/mynah-cli/server"
)

// MynahTaskType defines the type of a given task
type MynahTaskType string

//defines a key in context
type contextKey string

const (
	UploadTask          MynahTaskType = "mynah::upload"
	CreateICDatasetTask MynahTaskType = "mynah::ic::dataset::create"
)

// create new task data structs by type identifier
var taskConstructor = map[MynahTaskType]func() MynahTaskData{
	UploadTask:          func() MynahTaskData { return &MynahUploadTask{} },
	CreateICDatasetTask: func() MynahTaskData { return &MynahCreateICDatasetTask{} },
}

// MynahTaskId is the unique id for a task
type MynahTaskId string

// MynahTaskContext defines context added by each task
type MynahTaskContext map[MynahTaskId]context.Context

// MynahTaskData executes a task
type MynahTaskData interface {
	// ExecuteTask execute the task, return any context
	ExecuteTask(*server.MynahClient, MynahTaskContext) (context.Context, error)
}

// MynahTask defines a task to execute
type MynahTask struct {
	//the identifier for this task (must be unique). This is used by other
	//tasks to reference the result of this task
	TaskId MynahTaskId `json:"task_id"`
	//the type of the task (used for parsing)
	TaskType MynahTaskType `json:"task_type"`
	//the task itself
	TaskData MynahTaskData `json:"task_data"`
}

// MynahTaskSet defines a set of tasks to execute
type MynahTaskSet struct {
	//the tasks to execute
	Tasks []MynahTask `json:"tasks"`
}
