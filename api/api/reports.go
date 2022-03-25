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

const icReportKey string = "report"
const icReportBadImagesKey = "bad_images"
const icReportClassKey = "class"

// icDiagnosisReportFilter filters a report based on the provided options
func icDiagnosisReportFilter(report *model.MynahICDiagnosisReport,
	classes UniqueSet,
	badImagesFilter bool) *model.MynahICDiagnosisReport {

	imageIds := make([]string, 0)
	imageData := make(map[string]*model.MynahICDiagnosisReportImageMetadata)
	breakdown := make(map[string]*model.MynahICDiagnosisReportBucket)

	//add images within the filtered classes
	for fileId, metadata := range report.ImageData {
		if (!badImagesFilter || (len(metadata.OutlierSets) > 0) || metadata.Mislabeled) && classes.Contains(metadata.Class) {
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
	return report
}

// get a report from a given project
func icDiagnosisReportView(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		reportId, ok := mux.Vars(request)[icReportKey]
		//get request params
		if !ok {
			log.Errorf("report request path missing %s key", icReportKey)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//parse query params (optional)
		if err := request.ParseForm(); err != nil {
			log.Errorf("failed to parse request parameters to filter report: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		classes := NewUniqueSet()

		//class filtering options
		if vals, ok := request.Form[icReportClassKey]; ok {
			classes.Union(vals...)
		}

		//get filter options
		badImagesFilter := request.Form.Get(icReportBadImagesKey) == "true"

		//get the report from the database
		if report, err := dbProvider.GetICDiagnosisReport(&reportId, user); err == nil {
			//filter the report
			if err := responseWriteJson(writer, icDiagnosisReportFilter(report, classes, badImagesFilter)); err != nil {
				log.Errorf("failed to write response as json: %s", err)
				writer.WriteHeader(http.StatusInternalServerError)
			}

		} else {
			log.Warnf("failed to get ic report with id %s: %s", reportId, err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
	})
}
