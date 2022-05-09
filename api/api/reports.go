// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gorilla/mux"
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

const versionKey = "version"

// get a report from a given dataset
func icProcessReportView(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		datasetId, ok := mux.Vars(request)[datasetIdKey]
		//get request params
		if !ok {
			log.Errorf("report request path missing %s key", datasetIdKey)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//get the report from the database
		if dataset, err := dbProvider.GetICDataset(model.MynahUuid(datasetId), user, "versions"); err == nil {
			//the dataset version to get report for
			version := dataset.LatestVersion

			//parse query params (optional)
			if err := request.ParseForm(); err != nil {
				log.Errorf("failed to parse request parameters to filter report: %s", err)
				writer.WriteHeader(http.StatusInternalServerError)
				return
			}

			//class filtering options
			if v, ok := request.Form[versionKey]; ok && len(v) > 0 {
				version = model.MynahDatasetVersionId(v[0])
			}

			if binDataId, ok := dataset.Reports[version]; ok {
				//request the binary data
				binData, err := dbProvider.GetBinObject(binDataId, user)
				if err != nil {
					log.Errorf("failed to get dataset %s report version %s:", datasetId, version, err)
					writer.WriteHeader(http.StatusBadRequest)
					return
				}

				//respond with the json contents
				writer.Header().Add("Content-Type", "application/json")

				if _, err := writer.Write(binData.Data); err != nil {
					log.Errorf("failed to write report as json: %s", err)
				}

			} else {
				log.Warnf("dataset %s does not have a report for version %s", datasetId, version)
				writer.WriteHeader(http.StatusBadRequest)
				return
			}

		} else {
			log.Warnf("failed to get ic report from dataset with id %s: %s", datasetId, err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
	})
}
