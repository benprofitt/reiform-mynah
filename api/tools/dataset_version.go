// Copyright (c) 2022 by Reiform. All Rights Reserved.

package tools

import (
	"fmt"
	"reiform.com/mynah/db"
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
	"strconv"
)

// MakeICDatasetVersion creates a new ic dataset version.
// If a latest version exists, the files it references are versioned so that they won't be modified.
// Note: the contents of the new version will be empty and not contain copies of a previous version
func MakeICDatasetVersion(dataset *model.MynahICDataset,
	user *model.MynahUser,
	storageProvider storage.StorageProvider,
	dbProvider db.DBProvider) (newVersion *model.MynahICDatasetVersion, previousVersion *model.MynahICDatasetVersion, err error) {

	newVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions)))

	//verify that the version id doesn't already exist
	if _, ok := dataset.Versions[newVersionId]; ok {
		return nil, nil, fmt.Errorf("failed to create new dataset version, version id %s already exists", newVersionId)
	}

	//get the previous version id
	latestVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions) - 1))

	//check if there is a previous version
	if latestVersion, ok := dataset.Versions[latestVersionId]; ok {
		previousVersion = latestVersion
		//version the files in the current "latest" so they won't be modified
		for fileId, fileData := range latestVersion.Files {
			//request the file
			if file, err := dbProvider.GetFile(fileId, user); err == nil {
				//generate a new sha1
				newId, err := storageProvider.GenerateSHA1Id(file)
				if err != nil {
					return nil, nil, fmt.Errorf("failed to generate new version id for file: %s", err)
				}

				//only copy the file to the new version id if the id does not already exist
				if _, ok := file.Versions[newId]; !ok {
					//copy the latest version of the file to an explicit SHA1 version
					if err := storageProvider.CopyFile(file, model.LatestVersionId, newId); err != nil {
						return nil, nil, fmt.Errorf("failed to version from current latest dataset version: %s", err)
					}
				}

				//set the new id
				fileData.ImageVersionId = newId

			} else {
				return nil, nil, fmt.Errorf("failed to version from current latest dataset version: %s", err)
			}
		}
	}

	//create a new version
	dataset.Versions[newVersionId] = &model.MynahICDatasetVersion{
		Files: make(map[model.MynahUuid]*model.MynahICDatasetFile),
	}

	//return ptr to new version
	return dataset.Versions[newVersionId], previousVersion, nil
}

// MakeODDatasetVersion creates a new od dataset version
// If a latest version exists, the files it references are versioned so that they won't be modified.
// Note: the contents of the new version will be empty and not contain copies of a previous version
func MakeODDatasetVersion(dataset *model.MynahODDataset,
	user *model.MynahUser,
	storageProvider storage.StorageProvider,
	dbProvider db.DBProvider) (newVersion *model.MynahODDatasetVersion, previousVersion *model.MynahODDatasetVersion, err error) {

	newVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions)))

	//verify that the version id doesn't already exist
	if _, ok := dataset.Versions[newVersionId]; ok {
		return nil, nil, fmt.Errorf("failed to create new dataset version, version id %s already exists", newVersionId)
	}

	//get the previous version id
	latestVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions) - 1))

	//check if there is a previous version
	if latestVersion, ok := dataset.Versions[latestVersionId]; ok {
		previousVersion = latestVersion
		//version the files in the current "latest" so they won't be modified
		for fileId, fileData := range latestVersion.Files {
			//request the file
			if file, err := dbProvider.GetFile(fileId, user); err == nil {
				//generate a new sha1
				newId, err := storageProvider.GenerateSHA1Id(file)
				if err != nil {
					return nil, nil, fmt.Errorf("failed to generate new version id for file: %s", err)
				}

				//only copy the file to the new version id if the id does not already exist
				if _, ok := file.Versions[newId]; !ok {
					//copy the latest version of the file to an explicit SHA1 version
					if err := storageProvider.CopyFile(file, model.LatestVersionId, newId); err != nil {
						return nil, nil, fmt.Errorf("failed to version from current latest dataset version: %s", err)
					}
				}

				//set the new id
				fileData.ImageVersionId = newId

			} else {
				return nil, nil, fmt.Errorf("failed to version from current latest dataset version: %s", err)
			}
		}
	}

	//create a new version
	dataset.Versions[newVersionId] = &model.MynahODDatasetVersion{
		Entities:     make(map[model.MynahUuid]*model.MynahODDatasetEntity),
		Files:        make(map[model.MynahUuid]*model.MynahODDatasetFile),
		FileEntities: make(map[string][]model.MynahUuid),
	}

	//return ptr to new version
	return dataset.Versions[newVersionId], previousVersion, nil
}
