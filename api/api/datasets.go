// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"fmt"
	"github.com/gorilla/mux"
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/tools"
	"strconv"
)

const datasetIdKey = "datasetid"
const datasetVersionStartKey = "from_version"
const datasetVersionEndKey = "to_version"

// icDatasetCreate creates a new dataset in the database
func icDatasetCreate(dbProvider db.DBProvider, storageProvider storage.StorageProvider) http.HandlerFunc {
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
			//create an initial version of the dataset
			if initialVersion, _, err := tools.MakeICDatasetVersion(dataset, user, storageProvider, dbProvider); err == nil {
				//add the file id -> class name mappings, use the latest version of the file
				for fileId, className := range req.Files {
					initialVersion.Files[fileId] = &model.MynahICDatasetFile{
						ImageVersionId:    model.LatestVersionId,
						CurrentClass:      className,
						OriginalClass:     className,
						ConfidenceVectors: make(model.ConfidenceVectors, 0),
						Projections:       make(map[string][]int),
					}
				}

				//TODO do we want to cut explicit versions for these files? If they're updated in a different
				//dataset the changes will be reflected in this one
			} else {
				return fmt.Errorf("failed to create initial dataset version: %s", err)
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

// icDatasetFilterVersion filters a dataset based on the version range values. If neither were provided,
// only returns the latest version
func icDatasetFilterVersion(dataset *model.MynahICDataset, fromVersionParam, toVersionParam string, latestOnly bool) error {
	filteredVersions := make(map[model.MynahDatasetVersionId]*model.MynahICDatasetVersion)

	//by default, start with latest
	fromVersionNum := len(dataset.Versions) - 1

	if (len(fromVersionParam) > 0) && !latestOnly {
		if val, err := strconv.Atoi(fromVersionParam); err == nil {
			fromVersionNum = val
		} else {
			return fmt.Errorf("failed to parse dataset version range start: %s", err)
		}
	}

	//check if range requested
	if (len(toVersionParam) > 0) && !latestOnly {
		if toVersionNum, err := strconv.Atoi(toVersionParam); err == nil {
			for i := fromVersionNum; i < toVersionNum; i++ {
				currVersion := model.MynahDatasetVersionId(strconv.Itoa(fromVersionNum))

				if version, ok := dataset.Versions[currVersion]; ok {
					filteredVersions[currVersion] = version
				} else {
					//requested version not available
					return fmt.Errorf("dataset %s does not have version %s", dataset.Uuid, currVersion)
				}
			}

			//set the filtered list
			dataset.Versions = filteredVersions
			return nil

		} else {
			return fmt.Errorf("failed to parse dataset version range end: %s", err)
		}

	} else {
		//single version requested
		fromVersion := model.MynahDatasetVersionId(strconv.Itoa(fromVersionNum))

		if version, ok := dataset.Versions[fromVersion]; ok {
			filteredVersions[fromVersion] = version
			dataset.Versions = filteredVersions
			return nil
		} else {
			//requested version not available
			return fmt.Errorf("dataset %s does not have version %s", dataset.Uuid, fromVersion)
		}
	}
}

// odDatasetFilterVersion filters an od dataset
func odDatasetFilterVersion(dataset *model.MynahODDataset, fromVersionParam, toVersionParam string, latestOnly bool) error {
	filteredVersions := make(map[model.MynahDatasetVersionId]*model.MynahODDatasetVersion)

	//by default, start with latest
	fromVersionNum := len(dataset.Versions) - 1

	if (len(fromVersionParam) > 0) && !latestOnly {
		if val, err := strconv.Atoi(fromVersionParam); err == nil {
			fromVersionNum = val
		} else {
			return fmt.Errorf("failed to parse dataset version range start: %s", err)
		}
	}

	//check if range requested
	if (len(toVersionParam) > 0) && !latestOnly {
		if toVersionNum, err := strconv.Atoi(toVersionParam); err == nil {
			for i := fromVersionNum; i < toVersionNum; i++ {
				currVersion := model.MynahDatasetVersionId(strconv.Itoa(fromVersionNum))

				if version, ok := dataset.Versions[currVersion]; ok {
					filteredVersions[currVersion] = version
				} else {
					//requested version not available
					return fmt.Errorf("dataset %s does not have version %s", dataset.Uuid, currVersion)
				}
			}

			//set the filtered list
			dataset.Versions = filteredVersions
			return nil

		} else {
			return fmt.Errorf("failed to parse dataset version range end: %s", err)
		}

	} else {
		//single version requested
		fromVersion := model.MynahDatasetVersionId(strconv.Itoa(fromVersionNum))

		if version, ok := dataset.Versions[fromVersion]; ok {
			filteredVersions[fromVersion] = version
			dataset.Versions = filteredVersions
			return nil
		} else {
			//requested version not available
			return fmt.Errorf("dataset %s does not have version %s", dataset.Uuid, fromVersion)
		}
	}
}

// icDatasetGet gets an ic dataset by id. By default, returns the latest version only
func icDatasetGet(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		datasetId, ok := mux.Vars(request)[datasetIdKey]
		//get request params
		if !ok {
			log.Errorf("ic dataset request path missing %s key", datasetId)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//get the requested ic dataset
		dataset, err := dbProvider.GetICDataset(model.MynahUuid(datasetId), user)

		if err != nil {
			log.Errorf("failed to get ic dataset %s from database %s", datasetId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		//check for range params
		datasetFromVersion := request.Form.Get(datasetVersionStartKey)
		datasetToVersion := request.Form.Get(datasetVersionEndKey)

		//filter the dataset
		if err := icDatasetFilterVersion(dataset, datasetFromVersion, datasetToVersion, false); err != nil {
			log.Warnf("failed to filter dataset version(s): %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//write the response
		if err := responseWriteJson(writer, dataset); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

// odDatasetGet gets an od dataset by id. By default, returns the latest version only
func odDatasetGet(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		datasetId, ok := mux.Vars(request)[datasetIdKey]
		//get request params
		if !ok {
			log.Errorf("od dataset request path missing %s key", datasetId)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//get the requested od dataset
		dataset, err := dbProvider.GetODDataset(model.MynahUuid(datasetId), user)

		if err != nil {
			log.Errorf("failed to get od dataset %s from database %s", datasetId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		//check for range params
		datasetFromVersion := request.Form.Get(datasetVersionStartKey)
		datasetToVersion := request.Form.Get(datasetVersionEndKey)

		//filter the dataset
		if err := odDatasetFilterVersion(dataset, datasetFromVersion, datasetToVersion, false); err != nil {
			log.Warnf("failed to filter dataset version(s): %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//write the response
		if err := responseWriteJson(writer, dataset); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

// icDatasetList lists ic datasets
func icDatasetList(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		//list all ic datasets
		datasets, err := dbProvider.ListICDatasets(user)

		if err != nil {
			log.Errorf("failed to list ic datasets in database %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		for _, dataset := range datasets {
			//filter the get the latest version only
			if err = icDatasetFilterVersion(dataset, "", "", true); err != nil {
				log.Warnf("failed to filter ic dataset version(s): %s", err)
				writer.WriteHeader(http.StatusBadRequest)
				return
			}
		}

		//write the response
		if err := responseWriteJson(writer, &datasets); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

// odDatasetList lists od datasets
func odDatasetList(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		//list all od datasets
		datasets, err := dbProvider.ListODDatasets(user)

		if err != nil {
			log.Errorf("failed to list od datasets in database %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		for _, dataset := range datasets {
			//filter to get latest version only
			if err = odDatasetFilterVersion(dataset, "", "", true); err != nil {
				log.Warnf("failed to filter od dataset version(s): %s", err)
				writer.WriteHeader(http.StatusBadRequest)
				return
			}
		}

		//write the response
		if err := responseWriteJson(writer, &datasets); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

// allDatasetList lists datasets of all types
func allDatasetList(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		allDatasets := make([]model.MynahAbstractDataset, 0)

		icDatasets, err := dbProvider.ListICDatasets(user)

		if err != nil {
			log.Errorf("failed to list ic datasets in database %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//list all od datasets
		odDatasets, err := dbProvider.ListODDatasets(user)

		if err != nil {
			log.Errorf("failed to list od datasets in database %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		for _, dataset := range icDatasets {
			//filter to get latest version only
			if err = icDatasetFilterVersion(dataset, "", "", true); err != nil {
				log.Warnf("failed to filter ic dataset version(s): %s", err)
				writer.WriteHeader(http.StatusBadRequest)
				return
			}
			//add to the full collection
			allDatasets = append(allDatasets, dataset)
		}

		for _, dataset := range odDatasets {
			//filter to get latest version only
			if err = odDatasetFilterVersion(dataset, "", "", true); err != nil {
				log.Warnf("failed to filter od dataset version(s): %s", err)
				writer.WriteHeader(http.StatusBadRequest)
				return
			}
			//add to the full collection
			allDatasets = append(allDatasets, dataset)
		}

		//write the response
		if err := responseWriteJson(writer, &allDatasets); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
