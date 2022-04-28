// Copyright (c) 2022 by Reiform. All Rights Reserved.

package task

import (
	"context"
	"errors"
	"fmt"
	"reiform.com/mynah-cli/server"
	"reiform.com/mynah/api"
	"reiform.com/mynah/async"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"time"
)

// MynahICDiagnoseCleanTask defines the task of starting a clean/diagnose job
type MynahICDiagnoseCleanTask struct {
	//reference a dataset created previously
	FromExisting model.MynahUuid `json:"from_existing"`
	//reference dataset created in a previous task
	FromTask MynahTaskId `json:"from_task"`
	//whether to run the diagnose step
	Diagnose bool `json:"diagnose"`
	//whether to run the clean step
	Clean bool `json:"clean"`
	//the task polling frequency
	PollFrequency int `json:"poll_frequency"`
}

//run the diagnosis/clean job
func icDiagnoseClean(mynahServer *server.MynahClient, datasetId model.MynahUuid, diagnose, clean bool, pollFreq int) error {
	//make the request
	icDiagnoseCleanReq := api.ICCleanDiagnoseJobRequest{
		Diagnose:    diagnose,
		Clean:       clean,
		DatasetUuid: datasetId,
	}

	var response api.ICCleanDiagnoseJobResponse

	//make the request
	if err := mynahServer.ExecutePostJsonRequest("dataset/ic/diagnose_clean/start", &icDiagnoseCleanReq, &response); err != nil {
		return fmt.Errorf("failed to create ic diagnose/clean job: %s", err)
	}

	if pollFreq <= 0 {
		pollFreq = 10
	}

	//wait for completion
	for {
		var taskStatus api.TaskStatusResponse
		if err := mynahServer.ExecuteGetRequest(fmt.Sprintf("task/status/%s", response.TaskUuid), &taskStatus); err == nil {
			if taskStatus.TaskStatus == async.StatusCompleted {
				return nil
			} else if taskStatus.TaskStatus == async.StatusFailed {
				return errors.New("image classification diagnose/clean task failed")
			} else {
				log.Infof("image classification diagnose/clean task is %s", taskStatus.TaskStatus)
			}

		} else {
			return fmt.Errorf("failed to check status of ic diagnose/clean job: %s", err)
		}

		//wait to retry
		time.Sleep(time.Second * time.Duration(pollFreq))
	}
}

// ExecuteTask executes the create icdataset task
func (t MynahICDiagnoseCleanTask) ExecuteTask(mynahServer *server.MynahClient,
	tctx MynahTaskContext) (context.Context, error) {

	if (len(t.FromExisting) > 0) && (len(t.FromTask) > 0) {
		return nil, errors.New("failed to start diagnosis/cleaning, 'from_existing' and 'from_task' both set")
	}

	if len(t.FromExisting) > 0 {
		if err := icDiagnoseClean(mynahServer, t.FromExisting, t.Diagnose, t.Clean, t.PollFrequency); err != nil {
			return nil, err
		}

	} else if len(t.FromTask) > 0 {
		if datasetTaskData, ok := tctx[t.FromTask]; ok {
			if datasetId, ok := datasetTaskData.Value(CreatedICDatasetKey).(model.MynahUuid); ok {
				if err := icDiagnoseClean(mynahServer, datasetId, t.Diagnose, t.Clean, t.PollFrequency); err != nil {
					return nil, err
				}
			} else {
				return nil, fmt.Errorf("task %s does not have a dataset result", t.FromTask)
			}
		} else {
			return nil, fmt.Errorf("no such task: %s", t.FromTask)
		}

	} else {
		return nil, errors.New("failed to start diagnosis/cleaning, at least one of 'from_existing' or 'from_task' must be set")
	}

	//TODO report as output?

	return context.Background(), nil
}
