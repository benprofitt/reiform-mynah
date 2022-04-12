// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

// icDatasetCreate creates a new dataset in the database
func icDatasetCreate(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request (will be the owner)
		user := middleware.GetUserFromRequest(request)

		var req CreateICDatasetRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//create the dataset, set the name and the files
		dataset, err := dbProvider.CreateICDataset(user, func(dataset *model.MynahICDataset) error {
			dataset.DatasetName = req.Name
			dataset.Files = make(map[string]*model.MynahICDatasetFile)

			//add the file id -> class name mappings
			for fileId, className := range req.Files {
				dataset.Files[fileId] = &model.MynahICDatasetFile{
					CurrentClass:      className,
					OriginalClass:     className,
					ConfidenceVectors: make(model.ConfidenceVectors, 0),
					Projections:       make(map[string][]int),
				}
			}

			return nil
		})

		if err != nil {
			log.Errorf("failed to add new ic dataset to database %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//write the response
		if err := responseWriteJson(writer, dataset); err != nil {
			log.Warnf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
