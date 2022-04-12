// Copyright (c) 2022 by Reiform. All Rights Reserved.

package task

import (
	"context"
	"reiform.com/mynah-cli/server"
)

// MynahCreateICDatasetTask defines the task of creating an image classification dataset
type MynahCreateICDatasetTask struct {
	//reference files by id already in mynah
	FromExisting []string `json:"from_existing"`
	//reference files uploaded in previous tasks
	FromTasks []MynahTaskId `json:"from_tasks"`
	//TODO figure out class mappings

	//TODO maybe use a regex for parsing the class
}

// ExecuteTask executes the create icdataset task
func (t MynahCreateICDatasetTask) ExecuteTask(mynahServer *server.MynahClient,
	tctx MynahTaskContext) (context.Context, error) {

	//TODO
	return context.Background(), nil
}
