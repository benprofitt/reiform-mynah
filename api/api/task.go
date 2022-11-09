// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gorilla/mux"
	"net/http"
	"reiform.com/mynah/async"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

const TaskIdKey = "taskid"

// GetAsyncTaskStatus get the status of a task
func GetAsyncTaskStatus(asyncProvider async.AsyncProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the user from context
		user := middleware.GetUserFromRequest(request)

		taskId, ok := mux.Vars(request)[TaskIdKey]
		//get request params
		if !ok {
			log.Errorf("task status request path missing %s key", taskId)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//get the status of the task from the async engine
		if stat, err := asyncProvider.GetAsyncTaskStatus(user, model.MynahUuid(taskId)); err == nil {
			//write the response
			if err := responseWriteJson(writer, stat); err != nil {
				log.Warnf("failed to write response as json: %s", err)
				writer.WriteHeader(http.StatusInternalServerError)
			}
		} else {
			log.Errorf("failed to get task status: %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
	})
}

// ListAsyncTasks async tasks owned by the user
func ListAsyncTasks(asyncProvider async.AsyncProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the user from context
		user := middleware.GetUserFromRequest(request)

		//list tasks and write response
		if err := responseWriteJson(writer, asyncProvider.ListAsyncTasks(user)); err != nil {
			log.Warnf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
