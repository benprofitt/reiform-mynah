// Copyright (c) 2022 by Reiform. All Rights Reserved.

package task

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"reiform.com/mynah-cli/server"
	"reiform.com/mynah-cli/utils"
	"reiform.com/mynah/model"
)

// MynahICReportTask defines the task of requesting an ic process report
type MynahICReportTask struct {
	//reference a dataset created previously
	FromExisting model.MynahUuid `json:"from_existing"`
	//reference dataset created in a previous task
	FromTask MynahTaskId `json:"from_task"`
	//write the report to a file
	ToFile string `json:"to_file"`
}

// download the report and save to the given path
func requestWriteFile(mynahServer *server.MynahClient, datasetId model.MynahUuid, tofile string) error {
	var responseBody model.MynahICDatasetReport
	if err := mynahServer.ExecuteGetRequest(fmt.Sprintf("dataset/ic/%s/report", datasetId), &responseBody); err != nil {
		return fmt.Errorf("failed to request the report for dataset %s: %s", datasetId, err)
	}

	//serialize and write to file
	jsonData, err := json.MarshalIndent(responseBody, "", " ")
	if err != nil {
		return fmt.Errorf("failed to serialize the report for dataset %s: %s", datasetId, err)
	}
	return ioutil.WriteFile(tofile, jsonData, 0600)
}

// ExecuteTask executes the create icdataset task
func (t MynahICReportTask) ExecuteTask(mynahServer *server.MynahClient,
	tctx MynahTaskContext) (context.Context, error) {

	if err := utils.OneOf(string(t.FromExisting), string(t.FromTask)); err != nil {
		return nil, fmt.Errorf("failed to request image classification report, 'from_existing' and 'from_task' both: %s", err)
	}

	if len(t.ToFile) == 0 {
		return nil, errors.New("image classification report output location not specified")
	}

	if len(t.FromExisting) > 0 {
		if err := requestWriteFile(mynahServer, t.FromExisting, t.ToFile); err != nil {
			return nil, err
		}
	} else {
		if taskData, ok := tctx[t.FromTask]; ok {
			if err := utils.IsAllowedTaskType(taskData.TaskType, CreateICDatasetTask, ICProcessTask); err != nil {
				return nil, fmt.Errorf("image classification report request references incompatible previous task: %s", err)
			}
			//get the dataset id from the context
			if datasetId, ok := taskData.Value(ICDatasetKey).(model.MynahUuid); ok {
				if err := requestWriteFile(mynahServer, datasetId, t.ToFile); err != nil {
					return nil, err
				}

			} else {
				return nil, fmt.Errorf("task %s does not have a dataset result", t.FromTask)
			}
		} else {
			return nil, fmt.Errorf("no such task: %s", t.FromTask)
		}
	}

	return context.Background(), nil
}
