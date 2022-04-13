// Copyright (c) 2022 by Reiform. All Rights Reserved.

package task

import (
	"context"
	"fmt"
	"reiform.com/mynah-cli/server"
	"reiform.com/mynah/api"
	"reiform.com/mynah/model"
)

// CreatedICProjectKey is the key for the create project result context
const CreatedICProjectKey contextKey = "CreatedICProject"

// MynahCreateICProjectTask defines the task of creating an image classification project
type MynahCreateICProjectTask struct {
	//create from existing datasets
	FromExisting []string `json:"from_existing"`
	//from create dataset tasks
	FromTasks []MynahTaskId `json:"from_tasks"`
	//the name for the project
	ProjectName string `json:"project_name"`
}

// ExecuteTask executes the create icdataset task
func (t MynahCreateICProjectTask) ExecuteTask(mynahServer *server.MynahClient,
	tctx MynahTaskContext) (context.Context, error) {

	for _, datasetTask := range t.FromTasks {
		if datasetData, ok := tctx[datasetTask]; ok {
			if datasetId, ok := datasetData.Value(CreatedICDatasetKey).(string); ok {
				t.FromExisting = append(t.FromExisting, datasetId)
			} else {
				return nil, fmt.Errorf("task %s does not have dataset id to add to dataset", datasetTask)
			}
		} else {
			return nil, fmt.Errorf("no such task: %s", datasetTask)
		}
	}

	projectRequest := api.CreateICProjectRequest{
		Name:     t.ProjectName,
		Datasets: t.FromExisting,
	}

	var icProjectResponse model.MynahICProject

	//make the request
	if err := mynahServer.ExecutePostJsonRequest("icproject/create", &projectRequest, &icProjectResponse); err != nil {
		return nil, fmt.Errorf("failed to create ic project: %s", err)
	}

	//add the id to the context
	return context.WithValue(context.Background(), CreatedICProjectKey, icProjectResponse.Uuid), nil
}
