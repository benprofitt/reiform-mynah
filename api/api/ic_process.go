// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reiform.com/mynah/async"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/pyimpl"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/tools"
)

// icProcessJob handle request to start a new async job
func icProcessJob(dbProvider db.DBProvider,
	asyncProvider async.AsyncProvider,
	pyImplProvider pyimpl.PyImplProvider,
	storageProvider storage.StorageProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		user := middleware.GetUserFromRequest(request)

		var req ICProcessJobRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json for ic process job: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//get the dataset
		dataset, err := dbProvider.GetICDataset(req.DatasetUuid, user)
		if err != nil {
			log.Warnf("failed to get dataset for ic process job: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//kick off async job
		taskId := asyncProvider.StartAsyncTask(user, func(model.MynahUuid) ([]byte, error) {
			//create a new version of the dataset to operate on
			newVersion, err := tools.MakeICDatasetVersion(dataset,
				user,
				storageProvider,
				dbProvider)
			if err != nil {
				return nil, fmt.Errorf("ic process task for dataset %s failed when creating new version: %s", req.DatasetUuid, err)
			}

			//start the python task
			err = pyImplProvider.ICProcessJob(user, req.DatasetUuid, newVersion, req.Tasks)
			if err != nil {
				return nil, fmt.Errorf("ic process task for dataset %s failed: %s", req.DatasetUuid, err)
			}

			//freeze the fileids so that the "latest" versions aren't modified be a different dataset before a new version of this one is frozen
			if err = tools.FreezeICDatasetFileVersions(newVersion, user, storageProvider, dbProvider); err != nil {
				return nil, fmt.Errorf("ic process for dataset %s failed when freezing file versions: %s", req.DatasetUuid, err)
			}

			//update the results in the database (will overwrite any changes made to versions col since task started)
			if err := dbProvider.UpdateICDataset(dataset, user, "versions"); err != nil {
				return nil, fmt.Errorf("ic process for dataset %s failed when updating in database: %s", req.DatasetUuid, err)
			}

			//will distribute report over any websocket connections
			return json.Marshal(newVersion.Report)
		})

		//respond with the task id
		response := ICProcessJobResponse{
			TaskUuid: taskId,
		}

		//write the response
		if err := responseWriteJson(writer, &response); err != nil {
			log.Warnf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
