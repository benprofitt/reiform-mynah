// Copyright (c) 2022 by Reiform. All Rights Reserved.

package task

import (
	"context"
	"errors"
	"fmt"
	"reiform.com/mynah-cli/server"
	"reiform.com/mynah/api"
	"reiform.com/mynah/model"
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
}

//run the diagnosis/clean job
func icDiagnoseClean(mynahServer *server.MynahClient, datasetId model.MynahUuid, diagnose, clean bool) error {
	expectWsMessages := 0
	if diagnose {
		expectWsMessages++
	}
	if clean {
		expectWsMessages++
	}

	resChan := make(chan server.WsRes)

	//start a websocket listener
	go mynahServer.WebsocketListener(expectWsMessages, resChan)

	//make the request
	icDiagnoseCleanReq := api.ICCleanDiagnoseJobRequest{
		Diagnose:    diagnose,
		Clean:       clean,
		DatasetUuid: datasetId,
	}

	//make the request
	if err := mynahServer.ExecutePostJsonRequest("dataset/ic/diagnose_clean/start", &icDiagnoseCleanReq, nil); err != nil {
		return fmt.Errorf("failed to create ic diagnose/clean job: %s", err)
	}

	for i := 0; i < expectWsMessages; i++ {
		wsRes := <-resChan
		if wsRes.Error != nil {
			return wsRes.Error
		}

		//TODO unpack result
	}

	return nil
}

// ExecuteTask executes the create icdataset task
func (t MynahICDiagnoseCleanTask) ExecuteTask(mynahServer *server.MynahClient,
	tctx MynahTaskContext) (context.Context, error) {

	if (len(t.FromExisting) > 0) && (len(t.FromTask) > 0) {
		return nil, errors.New("failed to start diagnosis/cleaning, 'from_existing' and 'from_task' both set")
	}

	if len(t.FromExisting) > 0 {
		if err := icDiagnoseClean(mynahServer, t.FromExisting, t.Diagnose, t.Clean); err != nil {
			return nil, err
		}

	} else if len(t.FromTask) > 0 {
		if datasetTaskData, ok := tctx[t.FromTask]; ok {
			if datasetId, ok := datasetTaskData.Value(CreatedICDatasetKey).(model.MynahUuid); ok {
				if err := icDiagnoseClean(mynahServer, datasetId, t.Diagnose, t.Clean); err != nil {
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
