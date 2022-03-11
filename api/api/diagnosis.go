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

//returns an async handler for starting a diagnosis job
func startICDiagnosisHandler(dbProvider db.DBProvider,
	pyImplProvider pyimpl.PyImplProvider,
	user *model.MynahUser,
	req *pyimpl.ICDiagnosisJobRequest) async.AsyncTaskHandler {
	//Note: we shouldn't modify the user that was passed in, over the course
	//of handling the async request the user data may become stale
	return func(string) ([]byte, error) {
		// start the job and return the response
		if res, err := pyImplProvider.ICDiagnosisJob(user, req); err == nil {
			//TODO extract data from the response
			return json.Marshal(res)
		} else {
			log.Warnf("ic diagnosis job returned error: %s", err)
			return nil, err
		}
	}
}

//create a new request body from the given data
func startICDiagnosisJobCreateReq(projectId string,
	datasets []*model.MynahICDataset,
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
			//create a mapping
			jobRequest.Dataset.ClassFiles[classInfo.CurrentClass] = make(map[string]pyimpl.ICDiagnosisJobRequestFile)

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
				}

				//add the class -> tmp file -> data mapping
				jobRequest.Dataset.ClassFiles[classInfo.CurrentClass][tmpPath] = pyimpl.ICDiagnosisJobRequestFile{
					Uuid:     fileId,
					Width:    width,
					Height:   height,
					Channels: channels,
				}
			}
		}
	}
	return &jobRequest, nil
}

//handle request to start a new async job
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

				//all files
				allFiles := make(map[string]*model.MynahFile)

				// add files
				for _, dataset := range icDatasets {
					for fileid, _ := range dataset.Files {
						fileUuidSet.Union(fileid)
					}
				}

				//request all of the referenced files
				if files, err := dbProvider.GetFiles(fileUuidSet.Vals(), user); err == nil {
					//get the temp path for each file
					fileTempPaths := make(map[string]string)
					for _, f := range files {
						allFiles[f.Uuid] = f

						if path, err := storageProvider.GetTmpPath(f, model.TagLatest); err == nil {
							fileTempPaths[f.Uuid] = path
						} else {
							log.Warnf("failed to get temporary path to file: %s", err)
							writer.WriteHeader(http.StatusInternalServerError)
							return
						}
					}

					req, err := startICDiagnosisJobCreateReq(icProject.Uuid, icDatasets, allFiles, fileTempPaths)
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
