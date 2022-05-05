// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gorilla/mux"
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/tools"
)

const icReportBadImagesKey = "bad_images"
const icReportClassKey = "class"

// icProcessReportFilter filters a report based on the provided options
func icProcessReportFilter(report *model.MynahICDatasetReport,
	classes tools.UniqueSet,
	badImagesFilter bool) {

	imageIds := make([]model.MynahUuid, 0)
	imageData := make(map[model.MynahUuid]*model.MynahICDatasetReportImageMetadata)
	breakdown := make(map[string]*model.MynahICDatasetReportBucket)

	//add images within the filtered classes
	for fileId, metadata := range report.ImageData {
		if (!badImagesFilter || (len(metadata.OutlierTasks) > 0)) && classes.Contains(metadata.Class) {
			//add this image as metadata
			imageIds = append(imageIds, fileId)
			imageData[fileId] = metadata
		}
	}

	//add class breakdowns within the filtered classes
	for className, bucket := range report.Breakdown {
		if classes.Contains(className) {
			breakdown[className] = bucket
		}
	}

	//set the filtered data
	report.ImageIds = imageIds
	report.ImageData = imageData
	report.Breakdown = breakdown
}

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

		//parse query params (optional)
		if err := request.ParseForm(); err != nil {
			log.Errorf("failed to parse request parameters to filter report: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		classes := tools.NewUniqueSet()

		//class filtering options
		if vals, ok := request.Form[icReportClassKey]; ok {
			classes.Union(vals...)
		}

		//get filter options
		badImagesFilter := request.Form.Get(icReportBadImagesKey) == "true"

		//get the report from the database
		if dataset, err := dbProvider.GetICDataset(model.MynahUuid(datasetId), user); err == nil {
			//get the latest version
			if version, err := tools.GetICDatasetLatest(dataset); err == nil {
				//filter the report and write to response
				icProcessReportFilter(version.Report, classes, badImagesFilter)
				if err := responseWriteJson(writer, version.Report); err != nil {
					log.Errorf("failed to write response as json: %s", err)
					writer.WriteHeader(http.StatusInternalServerError)
				}
			} else {
				log.Warnf("failed to get ic report from dataset with id %s: %s", datasetId, err)
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
