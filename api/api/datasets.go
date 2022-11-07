// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"archive/zip"
	"bytes"
	"fmt"
	"github.com/gorilla/mux"
	"net/http"
	"reiform.com/mynah/containers"
	"reiform.com/mynah/db"
	"reiform.com/mynah/impl"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/tools"
	"strconv"
	"time"
)

const datasetIdKey = "datasetid"
const datasetVersionKey = "version"
const datasetVersionStartKey = "from_version"
const datasetVersionEndKey = "to_version"

// sanitizeICDataset removes fields that are not consumed by the api
func sanitizeICDataset(dataset *model.MynahICDataset) {
	for _, versionData := range dataset.Versions {
		//task data isn't sensitive but the data is contained by the report
		//Note: we CANNOT omit this from json altogether since XORM uses the
		//default json serialization as well for DB storage
		versionData.TaskData = nil
	}
}

// icDatasetCreate creates a new dataset in the database
func icDatasetCreate(dbProvider db.DBProvider, storageProvider storage.StorageProvider, implProvider impl.ImplProvider) http.HandlerFunc {
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

		//request referenced files as a group
		fileIdSet := containers.NewUniqueSet[model.MynahUuid]()
		for fileId := range req.Files {
			fileIdSet.Union(fileId)
		}

		//request all files
		files, err := dbProvider.GetFiles(fileIdSet.Vals(), user)
		if err != nil {
			log.Errorf("failed request files for ic dataset creation: %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		// get the latest version of each file
		latestFiles, err := files.GetLatestVersions()
		if err != nil {
			log.Errorf("failed request files for ic dataset creation: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		// get the files locally
		err = storageProvider.GetStoredFiles(latestFiles, func(localFiles storage.MynahLocalFileSet) error {
			// extract metadata from the files
			return implProvider.BatchImageMetadata(user, localFiles)
		})
		if err != nil {
			log.Errorf("failed to record metadata for files for ic dataset creation: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		// commit the changes to the database
		if err := dbProvider.UpdateFiles(files, user, db.NewMynahDBColumns(model.VersionsColName)); err != nil {
			log.Errorf("failed to update files for ic dataset creation: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//create the dataset, set the name and the files
		dataset, err := dbProvider.CreateICDataset(user, func(dataset *model.MynahICDataset) error {
			dataset.DatasetName = req.Name
			//create an initial version of the dataset
			if initialVersion, err := tools.MakeICDatasetVersion(dataset); err == nil {
				//add the file id -> class name mappings, use the latest version of the file
				for fileId, className := range req.Files {
					initialVersion.Files[fileId] = model.NewICDatasetFile()
					initialVersion.Files[fileId].CurrentClass = className
					initialVersion.Files[fileId].OriginalClass = className

					// use the latest version metadata
					if fileData, ok := files[fileId]; ok {
						// use the latest version of the file
						latest, err := fileData.GetFileVersion(model.LatestVersionId)
						if err != nil {
							return err
						}
						initialVersion.Files[fileId].Mean = latest.Metadata.GetDefaultFloatSlice(model.MetadataMean, nil)
						initialVersion.Files[fileId].StdDev = latest.Metadata.GetDefaultFloatSlice(model.MetadataStddev, nil)
					} else {
						return fmt.Errorf("failed to load file %s for dataset creation: %s", fileId, err)
					}
				}
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

		sanitizeICDataset(dataset)

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
		dataset, err := dbProvider.GetICDataset(model.MynahUuid(datasetId), user, db.NewMynahDBColumns())

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

		sanitizeICDataset(dataset)

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

			sanitizeICDataset(dataset)
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

			sanitizeICDataset(dataset)

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

// export a dataset
func icDatasetExport(dbProvider db.DBProvider, storageProvider storage.StorageProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		datasetId, ok := mux.Vars(request)[datasetIdKey]
		//get request params
		if !ok {
			log.Errorf("ic dataset export request path missing %s key", datasetId)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//get the dataset to be exported
		dataset, err := dbProvider.GetICDataset(model.MynahUuid(datasetId), user, db.NewMynahDBColumns())
		if err != nil {
			log.Warnf("failed to get ic dataset %s from database for export: %s", datasetId, err)
			writer.WriteHeader(http.StatusNotFound)
			return
		}

		datasetVersionId := model.MynahDatasetVersionId(request.Form.Get(datasetVersionKey))
		if datasetVersionId == "" {
			//if no specific version provided, default to latest version
			datasetVersionId = dataset.LatestVersion
		}

		datasetVersion, ok := dataset.Versions[datasetVersionId]
		if !ok {
			log.Warnf("no such dataset version for dataset %s: %s", datasetId, datasetVersionId)
			writer.WriteHeader(http.StatusNotFound)
			return
		}

		//request the files that are referenced by this dataset version
		fileUuidSet := containers.NewUniqueSet[model.MynahUuid]()
		for fileId := range datasetVersion.Files {
			//add to set
			fileUuidSet.Union(fileId)
		}

		//request the files in this dataset as a group
		files, err := dbProvider.GetFiles(fileUuidSet.Vals(), user)
		if err != nil {
			log.Warnf("failed to request files tracked by dataset %s for dataset export: %s", datasetId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//for each file, map to its original name
		filenames := make(map[model.MynahUuid]string)
		for fileId, file := range files {
			filenames[fileId] = file.Name
		}

		archiveName := fmt.Sprintf("%s_%s", dataset.DatasetName, datasetVersionId)

		//create a new zip archive
		archive, err := storageProvider.CreateTempFile(archiveName)
		if err != nil {
			log.Warnf("failed to create new zip archive for dataset %s during export: %s", datasetId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		// remove the temp file after serving the contents
		defer func(storageProvider storage.StorageProvider, s string) {
			err := storageProvider.DeleteTempFile(s)
			if err != nil {
				log.Warnf("failed to temp zip archive after export: %s", err)
			}
		}(storageProvider, archiveName)

		zipWriter := zip.NewWriter(archive)

		//write dataset files to archive
		err = dataset.Format.Metadata.DatasetFileIterator(datasetVersion, filenames,
			func(fileId model.MynahUuid, fileVersionId model.MynahFileVersionId, filePath string) error {
				fileVersion, err := files[fileId].GetFileVersion(fileVersionId)
				if err != nil {
					return err
				}
				//get the file from the storage provider and write the contents to the zip archive
				return storageProvider.GetStoredFile(fileId, fileVersion, func(localFile storage.MynahLocalFile) error {
					return tools.WriteToZip(zipWriter, localFile.Path(), filePath)
				})
			})

		if err != nil {
			log.Warnf("failed to write files for dataset %s during export: %s", datasetId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//write additional artifacts to archive
		err = dataset.Format.Metadata.GenerateArtifacts(datasetVersion,
			func(fileContents []byte, filePath string) error {
				// write to the zip archive
				if err := tools.WriteReaderToZip(zipWriter, bytes.NewReader(fileContents), filePath); err != nil {
					return fmt.Errorf("failed to write artifact %s to zip archive: %s", filePath, err)
				}
				return nil
			})

		if err != nil {
			log.Warnf("failed to write additional artifacts for dataset %s during export: %s", datasetId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		if err = zipWriter.Close(); err != nil {
			log.Warnf("failed to close zip writer: %s", datasetId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		if err = archive.Close(); err != nil {
			log.Warnf("failed to close zip archive: %s", datasetId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//respond with the zip archive
		http.ServeContent(writer, request, archiveName, time.Unix(0, 0), archive)
	})
}
