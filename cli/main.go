// Copyright (c) 2022 by Reiform. All Rights Reserved.

package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"os"
	"reiform.com/mynah-cli/server"
	"reiform.com/mynah-cli/task"
	"reiform.com/mynah/log"
)

func main() {
	taskPtr := flag.String("task", "task.json", "task path")
	protoPtr := flag.String("proto", "http", "http or https")
	addressPtr := flag.String("address", "127.0.0.1:8080", "server address")
	jwtPtr := flag.String("jwt", "", "auth jwt")
	flag.Parse()

	taskFile, err := os.Open(*taskPtr)

	if err != nil {
		log.Fatalf("failed to load Mynah task: %s", err)
	}

	//load the task
	taskData, err := ioutil.ReadAll(taskFile)
	if err != nil {
		log.Fatalf("failed to read from Mynah task file: %s", err)
	}

	var taskSet task.MynahTaskSet

	if err = json.Unmarshal(taskData, &taskSet); err != nil {
		log.Fatalf("failed to parse Mynah task file: %s", err)
	}

	serverClient := server.NewMynahClient(*addressPtr, *jwtPtr, *protoPtr)

	//create the base context
	tctx := make(task.MynahTaskContext)

	for i, mynahTask := range taskSet.Tasks {
		log.Infof("starting task  (%d/%d): (%s) %s", i+1, len(taskSet.Tasks), mynahTask.TaskType, mynahTask.TaskId)

		//execute the task which updates the context
		if ctx, err := mynahTask.TaskData.ExecuteTask(serverClient, tctx); err == nil {
			//set the updated context
			tctx[mynahTask.TaskId] = &task.MynahTaskContextData{
				TaskType: mynahTask.TaskType,
				TaskCtx:  ctx,
			}

			log.Infof("completed task (%d/%d)", i+1, len(taskSet.Tasks))
		} else {
			log.Fatalf("task %d failed: %s", i+1, err)
		}
	}
}
