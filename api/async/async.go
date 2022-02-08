// Copyright (c) 2022 by Reiform. All Rights Reserved.

package async

import (
	"context"
	"github.com/google/uuid"
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
	user *model.MynahUser
	//the task handler
	handler AsyncTaskHandler
	//the id of the task
	uuid string
}

//maintain the async queue, process new items
type asyncEngine struct {
	//channel for accepting new tasks
	taskChan chan *asyncTask
	//wait group for workers
	waitGroup sync.WaitGroup
	//task completion context
	ctx context.Context
	//task completion function
	cancel context.CancelFunc
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
			log.Infof("started async task %s at timestamp %d", task.uuid, start)
			//run the task
			res, err := task.handler(task.user)
			//get the stop timestamp
			stop := time.Now().Unix()

			if err != nil {
				log.Errorf("async task %s failed at timestamp %d: %s", task.uuid, stop, err)
			} else {
				log.Infof("async task %s succeeded at timestamp %d", task.uuid, stop)
				//send to the websocket to respond to client
				wsProvider.Send(&task.uuid, res)
			}
		}
	}
}

//create a new async provider that writes results to the websocket server
func NewAsyncProvider(mynahSettings *settings.MynahSettings, wsProvider websockets.WebSocketProvider) *asyncEngine {
	e := asyncEngine{
		taskChan:  make(chan *asyncTask, mynahSettings.AsyncSettings.BufferSize),
		waitGroup: sync.WaitGroup{},
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

//accept a new task
func (a *asyncEngine) StartAsyncTask(user *model.MynahUser, handler AsyncTaskHandler) {
	//write a new task to the channel
	a.taskChan <- &asyncTask{
		user:    user,
		handler: handler,
		uuid:    uuid.New().String(),
	}
}

//close async provider gracefully, finish pending tasks
func (a *asyncEngine) Close() {
	defer close(a.taskChan)

	//signal goroutines
	a.cancel()

	log.Infof("waiting for running async tasks to complete")
	a.waitGroup.Wait()
}
