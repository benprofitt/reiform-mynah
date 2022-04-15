// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reiform.com/mynah/tools"
	"strconv"

	"reiform.com/mynah/async"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/pyimpl"
	"reiform.com/mynah/storage"
)

// icDiagnosisRecordStatistics records whether this file is bad/acceptable for this task
func icDiagnosisRecordStatistics(className string,
	fileId model.MynahUuid,
	taskStatisticsBucket map[string]map[model.MynahUuid]*model.MynahICDiagnosisReportBucket,
	outlier bool) {

	//check if the class has already been seen
	if _, ok := taskStatisticsBucket[className]; !ok {
		taskStatisticsBucket[className] = make(map[model.MynahUuid]*model.MynahICDiagnosisReportBucket)
	}

	//check if the file has already been seen
	if _, ok := taskStatisticsBucket[className][fileId]; !ok {
		taskStatisticsBucket[className][fileId] = &model.MynahICDiagnosisReportBucket{
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
func icDiagnosisUpdateDatasetReportFromFileset(fileSet map[string]map[model.MynahUuid]pyimpl.ICDiagnosisJobResponseFile,
	taskName string,
	newFileData map[model.MynahUuid]*model.MynahICDatasetFile,
	prevFileData map[model.MynahUuid]*model.MynahICDatasetFile,
	report *model.MynahICDiagnosisReport,
	taskStatisticsBucket map[string]map[model.MynahUuid]*model.MynahICDiagnosisReportBucket,
	outliers bool) error {

	for className, classMap := range fileSet {
		for fileId, responseFile := range classMap {
			if _, ok := newFileData[fileId]; !ok {
				//add this file to the dataset, use the latest version id
				newFileData[fileId] = &model.MynahICDatasetFile{
					ImageVersionId:    model.LatestVersionId, //new version of dataset will use most recent versions of files
					CurrentClass:      responseFile.CurrentClass,
					OriginalClass:     responseFile.OriginalClass,
					ConfidenceVectors: responseFile.ConfidenceVectors,
					Projections:       responseFile.Projections,
				}
			}
			//else: file added by a different task

			outlierSet := make([]string, 0)

			if outliers {
				outlierSet = []string{taskName}
			}

			mislabeledTask := taskName == pyimpl.MislabeledTaskName

			//update statistics for later inclusion in report
			icDiagnosisRecordStatistics(className, fileId, taskStatisticsBucket, outliers)

			//check if this file has been encountered yet
			if reportFileData, ok := report.ImageData[fileId]; !ok {
				//check for the image version id
				if prevFileData, ok := prevFileData[fileId]; ok {
					report.ImageIds = append(report.ImageIds, fileId)
					report.ImageData[fileId] = &model.MynahICDiagnosisReportImageMetadata{
						ImageVersionId: prevFileData.ImageVersionId, //report will reference specific version of file
						Class:          className,
						Mislabeled:     outliers && mislabeledTask,
						Point: model.MynahICDiagnosisReportPoint{
							X: 0, //TODO
							Y: 0,
						},
						OutlierSets: outlierSet,
					}
				} else {
					return fmt.Errorf("file %s does not exist in previous dataset version", fileId)
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
func icDiagnosisUpdateDatasetReport(jobResponse *pyimpl.ICDiagnosisJobResponse,
	newFileData map[model.MynahUuid]*model.MynahICDatasetFile,
	prevFileData map[model.MynahUuid]*model.MynahICDatasetFile,
	report *model.MynahICDiagnosisReport) error {

	//map from class -> map from fileid to count of bad/acceptable for each task
	taskStatisticsBuckets := make(map[string]map[model.MynahUuid]*model.MynahICDiagnosisReportBucket)

	for _, task := range jobResponse.Tasks {
		if err := icDiagnosisUpdateDatasetReportFromFileset(task.Datasets.Inliers.ClassFiles,
			task.Name,
			newFileData,
			prevFileData,
			report,
			taskStatisticsBuckets,
			false); err != nil {
			return fmt.Errorf("failed to update inliers for dataset and report for task %s: %s", task.Name, err)
		}
		if err := icDiagnosisUpdateDatasetReportFromFileset(task.Datasets.Outliers.ClassFiles,
			task.Name,
			newFileData,
			prevFileData,
			report,
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
		report.Breakdown[className] = &model.MynahICDiagnosisReportBucket{
			Bad:        bad,
			Acceptable: acceptable,
		}
	}
	return nil
}

// startICDiagnosisHandler returns an async handler for starting a diagnosis job
func startICDiagnosisHandler(dbProvider db.DBProvider,
	storageProvider storage.StorageProvider,
	pyImplProvider pyimpl.PyImplProvider,
	user *model.MynahUser,
	req *pyimpl.ICDiagnosisJobRequest) async.AsyncTaskHandler {
	//Note: we shouldn't modify the user that was passed in, over the course
	//of handling the async request the user data may become stale
	return func(model.MynahUuid) ([]byte, error) {
		// start the job and return the response
		if res, err := pyImplProvider.ICDiagnosisJob(user, req); err == nil {
			//request the dataset to update
			dataset, err := dbProvider.GetICDataset(res.DatasetUuid, user)

			if err != nil {
				return nil, fmt.Errorf("failed to create report from ic diagnosis job: %s", err)
			}

			//create a new report based on the response
			report, err := dbProvider.CreateICDiagnosisReport(user, func(report *model.MynahICDiagnosisReport) error {
				//create a new version of the dataset
				if newVersion, previousVersion, err := tools.MakeICDatasetVersion(dataset, user, storageProvider, dbProvider); err == nil {
					//set the dataset file data and report data based on the diagnosis run
					if err := icDiagnosisUpdateDatasetReport(res, newVersion.Files, previousVersion.Files, report); err != nil {
						return err
					}
				} else {
					return err
				}
				//update the dataset
				return dbProvider.UpdateICDataset(dataset, user, "versions")
			})

			if err != nil {
				return nil, fmt.Errorf("failed to create report from ic diagnosis job: %s", err)
			}

			//send the report via any websocket connections
			return json.Marshal(report)
		} else {
			return nil, fmt.Errorf("ic diagnosis job returned error: %s", err)
		}
	}
}

// startICDiagnosisJobCreateReq create a new request body from the given data
func startICDiagnosisJobCreateReq(datasetId model.MynahUuid,
	dataset *model.MynahICDatasetVersion,
	files map[model.MynahUuid]*model.MynahFile,
	fileTmpPaths map[model.MynahUuid]string) (*pyimpl.ICDiagnosisJobRequest, error) {
	//build the job request
	jobRequest := pyimpl.ICDiagnosisJobRequest{
		DatasetUuid: datasetId,
	}
	jobRequest.Dataset.Classes = make([]string, 0)
	jobRequest.Dataset.ClassFiles = make(map[string]map[string]pyimpl.ICDiagnosisJobRequestFile)

	for fileId, classInfo := range dataset.Files {
		//create a mapping for this class if one doesn't exist
		if _, ok := jobRequest.Dataset.ClassFiles[classInfo.CurrentClass]; !ok {
			jobRequest.Dataset.ClassFiles[classInfo.CurrentClass] = make(map[string]pyimpl.ICDiagnosisJobRequestFile)
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
			jobRequest.Dataset.ClassFiles[classInfo.CurrentClass][tmpPath] = pyimpl.ICDiagnosisJobRequestFile{
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

// startICDiagnosisJob handle request to start a new async job
func startICDiagnosisJob(dbProvider db.DBProvider,
	asyncProvider async.AsyncProvider,
	pyImplProvider pyimpl.PyImplProvider,
	storageProvider storage.StorageProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		user := middleware.GetUserFromRequest(request)

		var req StartDiagnosisJobRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//get the dataset
		if icDataset, err := dbProvider.GetICDataset(req.DatasetUuid, user); err == nil {
			latestVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(icDataset.Versions) - 1))
			//use the latest version of the dataset
			if datasetVersion, ok := icDataset.Versions[latestVersionId]; ok {
				fileUuidSet := tools.NewUniqueSet()
				for fileId := range datasetVersion.Files {
					//add to set
					fileUuidSet.UuidsUnion(fileId)
				}

				//request all the files in the dataset
				if files, err := dbProvider.GetFiles(fileUuidSet.UuidVals(), user); err == nil {
					//get the temp path for each file
					fileTempPaths := make(map[model.MynahUuid]string)

					for fileId := range datasetVersion.Files {
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

					req, err := startICDiagnosisJobCreateReq(icDataset.Uuid, datasetVersion, files, fileTempPaths)
					if err != nil {
						log.Warnf("failed to create ic diagnosis job: %s", err)
						writer.WriteHeader(http.StatusInternalServerError)
						return
					}

					//kick off async job
					asyncProvider.StartAsyncTask(user,
						startICDiagnosisHandler(dbProvider, storageProvider, pyImplProvider, user, req))

				} else {
					log.Warnf("failed to request files: %s", err)
					writer.WriteHeader(http.StatusBadRequest)
					return
				}
			} else {
				log.Warnf("dataset does not have version: %d", latestVersionId)
				writer.WriteHeader(http.StatusBadRequest)
				return
			}

		} else {
			log.Warnf("failed to load dataset: %s", err)
			writer.WriteHeader(http.StatusUnauthorized)
			return
		}

		//create a response
		res := StartDiagnosisJobResponse{}

		//write the response
		if err := responseWriteJson(writer, &res); err != nil {
			log.Warnf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
