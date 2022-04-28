// Copyright (c) 2022 by Reiform. All Rights Reserved.

package async

import (
	"context"
	"fmt"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/websockets"
	"sync"
	"time"
)

//async task
type asyncTask struct {
	//the user who owns this task
	userUuid model.MynahUuid
	//the task handler
	handler AsyncTaskHandler
	//the id of the task
	taskUuid model.MynahUuid
}

//the current status of a task
type taskStatus struct {
	//the uuid of the user who has permission to view
	userUuid model.MynahUuid
	//the status
	stat MynahAsyncTaskStatus
}

//maintain the async queue, process new items
type asyncEngine struct {
	//channel for accepting new tasks
	taskChan chan *asyncTask
	//wait group for workers
	waitGroup sync.WaitGroup
	//task status lookup
	taskLookup map[model.MynahUuid]*taskStatus
	//mutex on task lookup
	taskLock sync.RWMutex
	//task completion context
	ctx context.Context
	//task completion function
	cancel context.CancelFunc
}

// setTaskStatus sets the status of a given task
func (a *asyncEngine) setTaskStatus(taskId model.MynahUuid, newStat MynahAsyncTaskStatus) {
	a.taskLock.Lock()
	defer a.taskLock.Unlock()
	if stat, ok := a.taskLookup[taskId]; ok {
		stat.stat = newStat
	} else {
		log.Warnf("unable to set status for task %s: does not exist", taskId)
	}
}

//execute tasks
func (a *asyncEngine) taskRunner(wsProvider websockets.WebSocketProvider) {
	defer a.waitGroup.Done()
	for {
		select {
		case <-a.ctx.Done():
			return

		case task := <-a.taskChan:
			start := time.Now().Unix()
			log.Infof("started async task %s at timestamp %d", task.taskUuid, start)
			a.setTaskStatus(task.taskUuid, StatusRunning)
			//run the task
			res, err := task.handler(task.userUuid)
			//get the stop timestamp
			stop := time.Now().Unix()

			if err != nil {
				a.setTaskStatus(task.taskUuid, StatusFailed)
			} else {
				a.setTaskStatus(task.taskUuid, StatusCompleted)
			}

			if err != nil {
				log.Errorf("async task %s failed at timestamp %d: %s", task.taskUuid, stop, err)
			} else {
				log.Infof("async task %s succeeded at timestamp %d", task.taskUuid, stop)
				if res != nil {
					//send to the websocket to respond to client
					wsProvider.Send(task.userUuid, res)
				}
			}
		}
	}
}

// NewAsyncProvider create a new async provider that writes results to the websocket server
func NewAsyncProvider(mynahSettings *settings.MynahSettings, wsProvider websockets.WebSocketProvider) *asyncEngine {
	e := asyncEngine{
		taskChan:   make(chan *asyncTask, mynahSettings.AsyncSettings.BufferSize),
		waitGroup:  sync.WaitGroup{},
		taskLookup: make(map[model.MynahUuid]*taskStatus),
	}
	workers := 1
	if mynahSettings.AsyncSettings.Workers > workers {
		workers = mynahSettings.AsyncSettings.Workers
	}

	//set the wait count
	e.waitGroup.Add(workers)

	//set the task completion context
	e.ctx, e.cancel = context.WithCancel(context.Background())

	log.Infof("starting async pool with %d workers", workers)
	for i := 0; i < workers; i++ {
		go e.taskRunner(wsProvider)
	}

	return &e
}

// StartAsyncTask accept a new task
func (a *asyncEngine) StartAsyncTask(user *model.MynahUser, handler AsyncTaskHandler) model.MynahUuid {
	taskUuid := model.NewMynahUuid()
	//write a new task to the channel
	a.taskChan <- &asyncTask{
		userUuid: user.Uuid,
		handler:  handler,
		taskUuid: taskUuid,
	}

	a.taskLock.Lock()
	defer a.taskLock.Unlock()

	//mark the task as pending
	a.taskLookup[taskUuid] = &taskStatus{
		userUuid: user.Uuid,
		stat:     StatusPending,
	}

	return taskUuid
}

// GetAsyncTaskStatus gets the status of a task
func (a *asyncEngine) GetAsyncTaskStatus(user *model.MynahUser, taskId model.MynahUuid) (MynahAsyncTaskStatus, error) {
	a.taskLock.RLock()
	defer a.taskLock.RUnlock()
	if stat, ok := a.taskLookup[taskId]; ok {
		if stat.userUuid == user.Uuid {
			return stat.stat, nil
		} else {
			return "", fmt.Errorf("user %s does not have permission to view task: %s", user.Uuid, taskId)
		}

	} else {
		return "", fmt.Errorf("no such task: %s", taskId)
	}
}

// Close close async provider gracefully, finish pending tasks
func (a *asyncEngine) Close() {
	defer close(a.taskChan)

	//signal goroutines
	a.cancel()

	log.Infof("waiting for running async tasks to complete")
	a.waitGroup.Wait()
}
