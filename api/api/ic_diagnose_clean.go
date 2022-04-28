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

// data used in diagnosis/cleaning
type icDiagnoseCleanContext struct {
	dbProvider      db.DBProvider
	storageProvider storage.StorageProvider
	pyImplProvider  pyimpl.PyImplProvider
	asyncProvider   async.AsyncProvider
	user            *model.MynahUser
	datasetId       model.MynahUuid
	datasetVersion  *model.MynahICDatasetVersion
	files           map[model.MynahUuid]*model.MynahFile
	fileTmpPaths    map[model.MynahUuid]string
}

// icDiagnosisRecordStatistics records whether this file is bad/acceptable for this task
func icDiagnosisRecordStatistics(className string,
	fileId model.MynahUuid,
	taskStatisticsBucket map[string]map[model.MynahUuid]*model.MynahICDatasetReportBucket,
	outlier bool) {

	//check if the class has already been seen
	if _, ok := taskStatisticsBucket[className]; !ok {
		taskStatisticsBucket[className] = make(map[model.MynahUuid]*model.MynahICDatasetReportBucket)
	}

	//check if the file has already been seen
	if _, ok := taskStatisticsBucket[className][fileId]; !ok {
		taskStatisticsBucket[className][fileId] = &model.MynahICDatasetReportBucket{
			Bad:        0,
			Acceptable: 0,
		}
	}

	if outlier {
		taskStatisticsBucket[className][fileId].Bad++
	} else {
		taskStatisticsBucket[className][fileId].Acceptable++
	}
}

// icDiagnosisUpdateDatasetReportFromFileset takes either the inliers or outliers and adds to the report/dataset
func icDiagnosisUpdateDatasetReportFromFileset(fileSet map[string]map[model.MynahUuid]pyimpl.ICDiagnoseCleanJobResponseFile,
	taskName string,
	datasetVersion *model.MynahICDatasetVersion,
	taskStatisticsBucket map[string]map[model.MynahUuid]*model.MynahICDatasetReportBucket,
	outliers bool) error {

	for className, classMap := range fileSet {
		for fileId, responseFile := range classMap {
			//update the file data
			if existingFile, ok := datasetVersion.Files[fileId]; ok {
				existingFile.CurrentClass = responseFile.CurrentClass
				existingFile.OriginalClass = responseFile.OriginalClass
				existingFile.ConfidenceVectors = responseFile.ConfidenceVectors
				existingFile.Projections = responseFile.Projections

			} else {
				return fmt.Errorf("ic diagnose/clean returned an unknown fileid: %s", fileId)
			}

			outlierSet := make([]string, 0)

			if outliers {
				outlierSet = []string{taskName}
			}

			mislabeledTask := taskName == pyimpl.MislabeledTaskName

			//update statistics for later inclusion in report
			icDiagnosisRecordStatistics(className, fileId, taskStatisticsBucket, outliers)

			//check if this file has been encountered yet
			if reportFileData, ok := datasetVersion.Report.ImageData[fileId]; !ok {
				datasetVersion.Report.ImageIds = append(datasetVersion.Report.ImageIds, fileId)
				datasetVersion.Report.ImageData[fileId] = &model.MynahICDatasetReportImageMetadata{
					ImageVersionId: model.LatestVersionId, //when a new version of the dataset is created, this will be updated
					Class:          className,
					Mislabeled:     outliers && mislabeledTask,
					Point: model.MynahICDatasetReportPoint{
						X: 0, //TODO
						Y: 0,
					},
					OutlierSets: outlierSet,
				}
			} else {
				reportFileData.OutlierSets = append(reportFileData.OutlierSets, outlierSet...)
				reportFileData.Mislabeled = reportFileData.Mislabeled || (outliers && mislabeledTask)
			}
		}
	}

	return nil
}

// icDiagnosisUpdateDatasetReport update the dataset, report based on the diagnosis results
func icDiagnosisUpdateDatasetReport(jobResponse *pyimpl.ICDiagnoseCleanJobResponse,
	datasetVersion *model.MynahICDatasetVersion) error {

	//map from class -> map from fileid to count of bad/acceptable for each task
	taskStatisticsBuckets := make(map[string]map[model.MynahUuid]*model.MynahICDatasetReportBucket)

	for _, task := range jobResponse.Tasks {
		if err := icDiagnosisUpdateDatasetReportFromFileset(task.Datasets.Inliers.ClassFiles,
			task.Name,
			datasetVersion,
			taskStatisticsBuckets,
			false); err != nil {
			return fmt.Errorf("failed to update inliers for dataset and report for task %s: %s", task.Name, err)
		}
		if err := icDiagnosisUpdateDatasetReportFromFileset(task.Datasets.Outliers.ClassFiles,
			task.Name,
			datasetVersion,
			taskStatisticsBuckets,
			true); err != nil {
			return fmt.Errorf("failed to update outliers for dataset and report for task %s: %s", task.Name, err)
		}
	}

	//add statistics for _all_ tasks into report
	for className, classMap := range taskStatisticsBuckets {
		bad := 0
		acceptable := 0

		for _, fileBucket := range classMap {
			//only consider this file acceptable if it isn't "bad" in any tasks
			if fileBucket.Bad == 0 {
				acceptable++
			} else {
				//this file is "bad" for at least one task
				bad++
			}
		}

		//add this class breakdown
		datasetVersion.Report.Breakdown[className] = &model.MynahICDatasetReportBucket{
			Bad:        bad,
			Acceptable: acceptable,
		}
	}
	return nil
}

// startICDiagnosisJobCreateReq create a new request body from the given data
func icDiagnoseJobRequest(datasetId model.MynahUuid,
	datasetVersion *model.MynahICDatasetVersion,
	files map[model.MynahUuid]*model.MynahFile,
	fileTmpPaths map[model.MynahUuid]string) (*pyimpl.ICDiagnoseCleanJobRequest, error) {
	//build the job request
	jobRequest := pyimpl.ICDiagnoseCleanJobRequest{
		DatasetUuid: datasetId,
	}
	jobRequest.Dataset.Classes = make([]string, 0)
	jobRequest.Dataset.ClassFiles = make(map[string]map[string]pyimpl.ICDiagnoseCleanJobRequestFile)

	for fileId, classInfo := range datasetVersion.Files {
		//create a mapping for this class if one doesn't exist
		if _, ok := jobRequest.Dataset.ClassFiles[classInfo.CurrentClass]; !ok {
			jobRequest.Dataset.ClassFiles[classInfo.CurrentClass] = make(map[string]pyimpl.ICDiagnoseCleanJobRequestFile)
		}

		//find the temp path for this file
		if tmpPath, found := fileTmpPaths[fileId]; found {
			//extract file metadata
			var channels int64
			var width int64
			var height int64

			//check for metadata
			if fileData, found := files[fileId]; found {
				//assume the latest version of the metadata
				if version, found := fileData.Versions[model.LatestVersionId]; found {
					channels = version.Metadata.GetDefaultInt(model.MetadataChannels, 0)
					width = version.Metadata.GetDefaultInt(model.MetadataWidth, 0)
					height = version.Metadata.GetDefaultInt(model.MetadataHeight, 0)
				} else {
					log.Warnf("unable to find file %s metadata using version id: %s",
						fileId, model.LatestVersionId)

					//attempt to get original
					if original, found := fileData.Versions[model.OriginalVersionId]; found {
						channels = original.Metadata.GetDefaultInt(model.MetadataChannels, 0)
						width = original.Metadata.GetDefaultInt(model.MetadataWidth, 0)
						height = original.Metadata.GetDefaultInt(model.MetadataHeight, 0)
					} else {
						return nil, fmt.Errorf("unable to find file %s metadata using version versionId: %s",
							fileId, model.OriginalVersionId)
					}
				}
			} else {
				return nil, fmt.Errorf("dataset file not found in files set for dataset: %s", datasetId)
			}

			//add the class -> tmp file -> data mapping
			jobRequest.Dataset.ClassFiles[classInfo.CurrentClass][tmpPath] = pyimpl.ICDiagnoseCleanJobRequestFile{
				Uuid:     fileId,
				Width:    width,
				Height:   height,
				Channels: channels,
			}

		} else {
			return nil, fmt.Errorf("file not found in downloaded files set for dataset: %s", datasetId)
		}
	}
	return &jobRequest, nil
}

// icDiagnoseCleanAsyncJob starts an async job
func (c icDiagnoseCleanContext) icDiagnoseCleanAsyncJob(diagnose, clean bool) async.AsyncTaskHandler {
	//Note: we shouldn't modify the user/dataset version that was passed in, over the course
	//of handling the async request the user data may become stale
	return func(model.MynahUuid) ([]byte, error) {
		//create a job request
		req, err := icDiagnoseJobRequest(c.datasetId, c.datasetVersion, c.files, c.fileTmpPaths)
		if err != nil {
			return nil, fmt.Errorf("ic diagnose/clean job failed when creating job request: %s", err)
		}

		//start the job
		res, err := c.pyImplProvider.ICDiagnoseCleanJob(c.user, req)
		if err != nil {
			return nil, fmt.Errorf("ic diagnose/clean job failed when starting ic diagnosis job: %s", err)
		}

		//request the dataset to make updates
		dataset, err := c.dbProvider.GetICDataset(res.DatasetUuid, c.user)
		if err != nil {
			return nil, fmt.Errorf("ic diagnose/clean job failed when requesting dataset to update: %s", err)
		}

		latestVersion, err := tools.GetICDatasetLatest(dataset)
		if err != nil {
			return nil, fmt.Errorf("ic diagnose/clean job failed: %s", err)
		}

		//create a new report
		latestVersion.Report = model.NewMynahICDatasetReport()

		//make changes to the report, dataset file mappings
		if err := icDiagnosisUpdateDatasetReport(res, latestVersion); err != nil {
			return nil, fmt.Errorf("ic diagnose/clean job failed when applying dataset changes: %s", err)
		}

		//update the dataset
		if err := c.dbProvider.UpdateICDataset(dataset, c.user, "versions"); err != nil {
			return nil, fmt.Errorf("ic diagnose/clean job failed when updating with result: %s", err)
		}

		//will distribute report over any websocket connections
		return json.Marshal(latestVersion.Report)
	}
}

// icDiagnoseCleanJob handle request to start a new async job
func icDiagnoseCleanJob(dbProvider db.DBProvider,
	asyncProvider async.AsyncProvider,
	pyImplProvider pyimpl.PyImplProvider,
	storageProvider storage.StorageProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		user := middleware.GetUserFromRequest(request)

		var req ICCleanDiagnoseJobRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json for ic diagnose/clean job: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		runDiagnosisStep := req.Diagnose
		runCleanStep := req.Clean

		if !runDiagnosisStep && !runCleanStep {
			log.Warnf("diagnosis and cleaning both disabled")
			http.Error(writer, "diagnosis and cleaning both disabled in request", http.StatusBadRequest)
			return
		}

		//get the dataset
		dataset, err := dbProvider.GetICDataset(req.DatasetUuid, user)
		if err != nil {
			log.Warnf("failed to get dataset for ic diagnose/clean job: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//create a new version of the dataset to operate on
		newVersion, err := tools.MakeICDatasetVersion(dataset,
			user,
			storageProvider,
			dbProvider)

		if err == nil {
			//load the files in the dataset
			fileUuidSet := tools.NewUniqueSet()
			for fileId := range newVersion.Files {
				//add to set
				fileUuidSet.UuidsUnion(fileId)
			}

			files, err := dbProvider.GetFiles(fileUuidSet.UuidVals(), user)
			if err != nil {
				log.Warnf("failed to load dataset %s files for ic diagnose/clean job: %s", dataset.Uuid, err)
				http.Error(writer, err.Error(), http.StatusInternalServerError)
				return
			}

			//get the temp path for each file
			fileTempPaths := make(map[model.MynahUuid]string)

			for fileId := range newVersion.Files {
				if f, ok := files[fileId]; ok {
					//get a path to this file, only use latest version id: other versions of file are immutable
					if path, err := storageProvider.GetTmpPath(f, model.LatestVersionId); err == nil {
						fileTempPaths[fileId] = path
					} else {
						log.Warnf("failed to get temporary path to file: %s", err)
						writer.WriteHeader(http.StatusInternalServerError)
						return
					}
				} else {
					log.Warnf("failed to get file: %s", err)
					writer.WriteHeader(http.StatusInternalServerError)
					return
				}
			}

			ctx := icDiagnoseCleanContext{
				dbProvider:      dbProvider,
				storageProvider: storageProvider,
				pyImplProvider:  pyImplProvider,
				asyncProvider:   asyncProvider,
				user:            user,
				datasetId:       req.DatasetUuid,
				datasetVersion:  newVersion,
				files:           files,
				fileTmpPaths:    fileTempPaths,
			}

			//update the dataset in the db with the new version
			if err := dbProvider.UpdateICDataset(dataset, user, "versions"); err != nil {
				log.Warnf("failed to update dataset %s with new version: %s", dataset.Uuid, err)
				http.Error(writer, err.Error(), http.StatusInternalServerError)
				return
			}

			//kick off async job
			taskId := asyncProvider.StartAsyncTask(user, ctx.icDiagnoseCleanAsyncJob(runDiagnosisStep, runCleanStep))

			//respond with the task id
			response := ICCleanDiagnoseJobResponse{
				TaskUuid: taskId,
			}

			//write the response
			if err := responseWriteJson(writer, &response); err != nil {
				log.Warnf("failed to write response as json: %s", err)
				writer.WriteHeader(http.StatusInternalServerError)
			}

		} else {
			log.Warnf("failed to create new version of dataset %s for ic diagnose/clean job: %s", dataset.Uuid, err)
			http.Error(writer, err.Error(), http.StatusInternalServerError)
			return
		}
	})
}
