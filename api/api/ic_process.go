// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"fmt"
	"net/http"
	"reiform.com/mynah/async"
	"reiform.com/mynah/db"
	"reiform.com/mynah/impl"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

// icProcessJob handle request to start a new async job
func icProcessJob(dbProvider db.DBProvider,
	asyncProvider async.AsyncProvider,
	implProvider impl.ImplProvider) middleware.HandlerFunc {
	return func(ctx *middleware.Context) {
		var req ICProcessJobRequest

		//attempt to parse the request body
		if err := ctx.ReadJson(&req); err != nil {
			ctx.Error(http.StatusBadRequest, "failed to parse json for ic process job: %s", err)
			return
		}

		//kick off async job
		taskId := asyncProvider.StartAsyncTask(ctx.User, func(model.MynahUuid) ([]byte, error) {
			//start the python task
			err := implProvider.ICProcessJob(ctx.User, req.DatasetUuid, req.Tasks)
			if err != nil {
				return nil, fmt.Errorf("ic process task for dataset %s failed: %s", req.DatasetUuid, err)
			}

			return nil, nil
		})

		//respond with the task id
		response := ICProcessJobResponse{
			TaskUuid: taskId,
		}

		//write the response
		if err := ctx.WriteJson(&response); err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to write response as json: %s", err)
			return
		}
	}
}
