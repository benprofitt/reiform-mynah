// Copyright (c) 2022 by Reiform. All Rights Reserved.

package async

import (
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/websockets"
	"testing"
	"time"
)

//setup and teardown
func TestMain(m *testing.M) {
	dirPath := "data"

	//create the base directory if it doesn't exist
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		log.Fatalf("failed to create directory: %s", dirPath)
	}

	//run tests
	exitVal := m.Run()

	//remove generated
	if err := os.RemoveAll(dirPath); err != nil {
		log.Errorf("failed to clean up after tests: %s", err)
	}

	os.Exit(exitVal)
}

//run tests on the async architecture
func TestAsync(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//init the websocket server
	wsProvider := websockets.NewWebSocketProvider(mynahSettings)
	defer wsProvider.Close()

	//init the async engine
	asyncProvider := NewAsyncProvider(mynahSettings, wsProvider)
	defer asyncProvider.Close()

	taskCount := 10

	user := model.MynahUser{
		Uuid: "test_async_user",
	}

	resultChan := make(chan int, taskCount)

	//add some tasks
	for i := 0; i < taskCount; i++ {
		//start tasks
		asyncProvider.StartAsyncTask(&user, func(uuid model.MynahUuid) ([]byte, error) {
			time.Sleep(time.Second * 1)
			resultChan <- 0
			return nil, nil
		})
	}

	//wait for tasks to complete
	for i := 0; i < taskCount; i++ {
		<-resultChan
	}
}

//test the status function
func TestAsyncStatus(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//init the websocket server
	wsProvider := websockets.NewWebSocketProvider(mynahSettings)
	defer wsProvider.Close()

	//init the async engine
	asyncProvider := NewAsyncProvider(mynahSettings, wsProvider)
	defer asyncProvider.Close()

	user := model.MynahUser{
		Uuid: "test_async_user",
	}

	taskChan := make(chan int)

	taskId := asyncProvider.StartAsyncTask(&user, func(uuid model.MynahUuid) ([]byte, error) {
		//wait to exit
		<-taskChan
		return nil, nil
	})

	time.Sleep(time.Second * 1)

	if status, err := asyncProvider.GetAsyncTaskStatus(&user, taskId); err == nil {
		if status != StatusRunning {
			t.Fatalf("expected task status to be %s but got %s", StatusRunning, status)
		}
	} else {
		t.Fatalf("query task status returned error: %s", err)
	}

	//cause the task to end
	taskChan <- 0

	time.Sleep(time.Second * 1)

	//get the status again
	if status, err := asyncProvider.GetAsyncTaskStatus(&user, taskId); err == nil {
		if status != StatusCompleted {
			t.Fatalf("expected task status to be %s but got %s", StatusCompleted, status)
		}
	} else {
		t.Fatalf("query task status returned error: %s", err)
	}
}
