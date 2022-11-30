// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"net/http"
	"reiform.com/mynah/async"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

const TaskIdKey = "taskid"

// getAsyncTaskStatus get the status of a task
func getAsyncTaskStatus(asyncProvider async.AsyncProvider) middleware.HandlerFunc {
	return func(ctx *middleware.Context) {
		taskId, ok := ctx.Vars()[TaskIdKey]
		//get request params
		if !ok {
			ctx.Error(http.StatusBadRequest, "task status request path missing %s key", taskId)
			return
		}

		stat, err := asyncProvider.GetAsyncTaskStatus(ctx.User, model.MynahUuid(taskId))
		if err != nil {
			ctx.Error(http.StatusBadRequest, "failed to get task status: %s", err)
			return
		}

		//write the response
		if err := ctx.WriteJson(stat); err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to write response as json: %s", err)
			return
		}
	}
}

// listAsyncTasks async tasks owned by the user
func listAsyncTasks(asyncProvider async.AsyncProvider) middleware.HandlerFunc {
	return func(ctx *middleware.Context) {
		if err := ctx.WriteJson(asyncProvider.ListAsyncTasks(ctx.User)); err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to write response as json: %s", err)
		}
	}
}
