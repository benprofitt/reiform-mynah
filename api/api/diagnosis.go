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
)

// icDiagnosisRecordStatistics records whether this file is bad/acceptable for this task
func icDiagnosisRecordStatistics(className,
	fileId string,
	taskStatisticsBucket map[string]map[string]*model.MynahICDiagnosisReportBucket,
	outlier bool) {

	//check if the class has already been seen
	if _, ok := taskStatisticsBucket[className]; !ok {
		taskStatisticsBucket[className] = make(map[string]*model.MynahICDiagnosisReportBucket)
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
func icDiagnosisUpdateDatasetReportFromFileset(fileSet map[string]map[string]pyimpl.ICDiagnosisJobResponseFile,
	taskName string,
	fileData map[string]*model.MynahICDatasetFile,
	report *model.MynahICDiagnosisReport,
	taskStatisticsBucket map[string]map[string]*model.MynahICDiagnosisReportBucket,
	outliers bool) {

	for className, classMap := range fileSet {
		for fileId, responseFile := range classMap {
			if _, ok := fileData[fileId]; !ok {
				//add this file to the dataset
				fileData[fileId] = &model.MynahICDatasetFile{
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
			if fileData, ok := report.ImageData[fileId]; !ok {
				report.ImageIds = append(report.ImageIds, fileId)
				report.ImageData[fileId] = &model.MynahICDiagnosisReportImageMetadata{
					Class:      className,
					Mislabeled: outliers && mislabeledTask,
					Point: model.MynahICDiagnosisReportPoint{
						X: 0, //TODO
						Y: 0,
					},
					OutlierSets: outlierSet,
				}
			} else {
				fileData.OutlierSets = append(fileData.OutlierSets, outlierSet...)
				fileData.Mislabeled = fileData.Mislabeled || (outliers && mislabeledTask)
			}
		}
	}
}

// icDiagnosisUpdateDatasetReport update the dataset, report based on the diagnosis results
func icDiagnosisUpdateDatasetReport(jobResponse *pyimpl.ICDiagnosisJobResponse,
	fileData map[string]*model.MynahICDatasetFile,
	report *model.MynahICDiagnosisReport) {

	//map from class -> map from fileid to count of bad/acceptable for each task
	taskStatisticsBuckets := make(map[string]map[string]*model.MynahICDiagnosisReportBucket)

	for _, task := range jobResponse.Tasks {
		icDiagnosisUpdateDatasetReportFromFileset(task.Datasets.Inliers.ClassFiles,
			task.Name,
			fileData,
			report,
			taskStatisticsBuckets,
			false)
		icDiagnosisUpdateDatasetReportFromFileset(task.Datasets.Outliers.ClassFiles,
			task.Name,
			fileData,
			report,
			taskStatisticsBuckets,
			true)
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
}

// startICDiagnosisHandler returns an async handler for starting a diagnosis job
func startICDiagnosisHandler(dbProvider db.DBProvider,
	pyImplProvider pyimpl.PyImplProvider,
	user *model.MynahUser,
	req *pyimpl.ICDiagnosisJobRequest) async.AsyncTaskHandler {
	//Note: we shouldn't modify the user that was passed in, over the course
	//of handling the async request the user data may become stale
	return func(string) ([]byte, error) {
		// start the job and return the response
		if res, err := pyImplProvider.ICDiagnosisJob(user, req); err == nil {
			//request the project to update
			project, err := dbProvider.GetICProject(&res.ProjectUuid, user)

			if err != nil {
				return nil, fmt.Errorf("failed to create report from ic diagnosis job: %s", err)
			}

			//create a new report based on the response
			report, err := dbProvider.CreateICDiagnosisReport(user, func(report *model.MynahICDiagnosisReport) error {

				//if there are more than 1 datasets in the project, merge them into a new dataset
				if len(project.Datasets) > 1 {
					//create a new dataset
					_, err := dbProvider.CreateICDataset(user, func(newDataset *model.MynahICDataset) error {
						//merge datasets
						if datasets, err := dbProvider.GetICDatasets(project.Datasets, user); err == nil {
							for _, dataset := range datasets {
								for fileId, fileData := range dataset.Files {
									newDataset.Files[fileId] = fileData
								}
							}
						} else {
							return err
						}

						//set the dataset file data and report data based on the diagnosis run
						icDiagnosisUpdateDatasetReport(res, newDataset.Files, report)

						project.Datasets = []string{newDataset.Uuid}

						//update the project
						return dbProvider.UpdateICProject(project, user, "datasets")
					})

					return err

				} else if len(project.Datasets) == 1 {
					//update existing
					if dataset, err := dbProvider.GetICDataset(&project.Datasets[0], user); err == nil {
						//set the dataset file data and report data based on the diagnosis run
						icDiagnosisUpdateDatasetReport(res, dataset.Files, report)

						//update the dataset
						if err = dbProvider.UpdateICDataset(dataset, user, "files"); err != nil {
							return err
						}

					} else {
						return err
					}
				} else {
					return fmt.Errorf("project %s does not have any associated datasets", project.Uuid)
				}

				return nil
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
func startICDiagnosisJobCreateReq(projectId string,
	datasets map[string]*model.MynahICDataset,
	files map[string]*model.MynahFile,
	fileTmpPaths map[string]string) (*pyimpl.ICDiagnosisJobRequest, error) {
	//build the job request
	jobRequest := pyimpl.ICDiagnosisJobRequest{
		ProjectUuid: projectId,
	}
	jobRequest.Dataset.Classes = make([]string, 0)
	jobRequest.Dataset.ClassFiles = make(map[string]map[string]pyimpl.ICDiagnosisJobRequestFile)

	//iterate over all project data
	for _, dataset := range datasets {
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
					if version, found := fileData.Versions[model.TagLatest]; found {
						channels = version.Metadata.GetDefaultInt(model.MetadataChannels, 0)
						width = version.Metadata.GetDefaultInt(model.MetadataWidth, 0)
						height = version.Metadata.GetDefaultInt(model.MetadataHeight, 0)
					} else {
						log.Warnf("unable to find file %s metadata using version tag: %s",
							fileId, model.TagLatest)

						//attempt to get original
						if original, found := fileData.Versions[model.TagOriginal]; found {
							channels = original.Metadata.GetDefaultInt(model.MetadataChannels, 0)
							width = original.Metadata.GetDefaultInt(model.MetadataWidth, 0)
							height = original.Metadata.GetDefaultInt(model.MetadataHeight, 0)
						} else {
							return nil, fmt.Errorf("unable to find file %s metadata using version tag: %s",
								fileId, model.TagOriginal)
						}
					}
				} else {
					return nil, fmt.Errorf("dataset file not found in files set for project: %s", projectId)
				}

				//add the class -> tmp file -> data mapping
				jobRequest.Dataset.ClassFiles[classInfo.CurrentClass][tmpPath] = pyimpl.ICDiagnosisJobRequestFile{
					Uuid:     fileId,
					Width:    width,
					Height:   height,
					Channels: channels,
				}

			} else {
				return nil, fmt.Errorf("dataset file not found in downloaded files set for project: %s", projectId)
			}
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

		var req startDiagnosisJobRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//request the project
		if icProject, err := dbProvider.GetICProject(&req.ProjectUuid, user); err == nil {
			//load referenced mynah datasets
			if icDatasets, err := dbProvider.GetICDatasets(icProject.Datasets, user); err == nil {
				//create a set for file uuids
				fileUuidSet := NewUniqueSet()

				// add files
				for _, dataset := range icDatasets {
					for fileid := range dataset.Files {
						fileUuidSet.Union(fileid)
					}
				}

				//request all of the referenced files
				if files, err := dbProvider.GetFiles(fileUuidSet.Vals(), user); err == nil {
					//get the temp path for each file
					fileTempPaths := make(map[string]string)
					for fileId, f := range files {
						if path, err := storageProvider.GetTmpPath(f, model.TagLatest); err == nil {
							fileTempPaths[fileId] = path
						} else {
							log.Warnf("failed to get temporary path to file: %s", err)
							writer.WriteHeader(http.StatusInternalServerError)
							return
						}
					}

					req, err := startICDiagnosisJobCreateReq(icProject.Uuid, icDatasets, files, fileTempPaths)
					if err != nil {
						log.Warnf("failed to create ic diagnosis job: %s", err)
						writer.WriteHeader(http.StatusInternalServerError)
						return
					}

					//kick off async job
					asyncProvider.StartAsyncTask(user,
						startICDiagnosisHandler(dbProvider, pyImplProvider, user, req))

				} else {
					log.Warnf("failed to load files: %s", err)
					writer.WriteHeader(http.StatusInternalServerError)
					return
				}
			} else {
				log.Warnf("failed to load datasets: %s", err)
				writer.WriteHeader(http.StatusInternalServerError)
				return
			}
		} else {
			log.Warnf("failed to load project: %s", err)
			writer.WriteHeader(http.StatusUnauthorized)
			return
		}

		//create a response
		res := startDiagnosisJobResponse{}

		//write the response
		if err := responseWriteJson(writer, &res); err != nil {
			log.Warnf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
