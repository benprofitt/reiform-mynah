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

//maintain the async queue, process new items
type asyncEngine struct {
	//channel for accepting new tasks
	taskChan chan *asyncTask
	//wait group for workers
	waitGroup sync.WaitGroup
	//task status lookup (map from user to map from task id)
	taskLookup map[model.MynahUuid]map[model.MynahUuid]*AsyncTaskData
	//mutex on task lookup
	taskLock sync.RWMutex
	//task completion context
	ctx context.Context
	//task completion function
	cancel context.CancelFunc
}

// setTaskStatus sets the status of a given task
func (a *asyncEngine) setTaskStatus(userUuid model.MynahUuid, taskId model.MynahUuid, newStat MynahAsyncTaskStatus) {
	a.taskLock.Lock()
	defer a.taskLock.Unlock()
	if userTasks, ok := a.taskLookup[userUuid]; ok {
		if stat, ok := userTasks[taskId]; ok {
			stat.TaskStatus = newStat
			return
		}
	}

	log.Warnf("unable to set status for task %s owned by user %s: does not exist", taskId, userUuid)
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
			a.setTaskStatus(task.userUuid, task.taskUuid, StatusRunning)
			//run the task
			res, err := task.handler(task.userUuid)
			//get the stop timestamp
			stop := time.Now().Unix()

			if err != nil {
				a.setTaskStatus(task.userUuid, task.taskUuid, StatusFailed)
			} else {
				a.setTaskStatus(task.userUuid, task.taskUuid, StatusCompleted)
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
		taskLookup: make(map[model.MynahUuid]map[model.MynahUuid]*AsyncTaskData),
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

	if _, ok := a.taskLookup[user.Uuid]; !ok {
		a.taskLookup[user.Uuid] = make(map[model.MynahUuid]*AsyncTaskData)
	}

	//mark the task as pending
	a.taskLookup[user.Uuid][taskUuid] = &AsyncTaskData{
		Started:    time.Now().Unix(),
		TaskId:     taskUuid,
		TaskStatus: StatusPending,
	}

	return taskUuid
}

// GetAsyncTaskStatus gets the status of a task
func (a *asyncEngine) GetAsyncTaskStatus(user *model.MynahUser, taskId model.MynahUuid) (*AsyncTaskData, error) {
	a.taskLock.RLock()
	defer a.taskLock.RUnlock()

	if userTasks, ok := a.taskLookup[user.Uuid]; ok {
		if stat, ok := userTasks[taskId]; ok {
			return stat, nil
		} else {
			return nil, fmt.Errorf("user %s does not have task: %s", user.Uuid, taskId)
		}
	} else {
		return nil, fmt.Errorf("user %s does not have any tasks", user.Uuid)
	}
}

// ListAsyncTasks lists the async tasks owned by a user
func (a *asyncEngine) ListAsyncTasks(user *model.MynahUser) (res []*AsyncTaskData) {
	a.taskLock.RLock()
	defer a.taskLock.RUnlock()

	res = make([]*AsyncTaskData, 0)

	// iterate over the users tasks, add status
	if userTasks, ok := a.taskLookup[user.Uuid]; ok {
		for _, task := range userTasks {
			res = append(res, task)
		}
	}

	return res
}

// Close async provider gracefully, finish pending tasks
func (a *asyncEngine) Close() {
	defer close(a.taskChan)

	//signal goroutines
	a.cancel()

	log.Infof("waiting for running async tasks to complete")
	a.waitGroup.Wait()
}
