// Copyright (c) 2022 by Reiform. All Rights Reserved.

package tools

import (
	"fmt"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
	"strconv"
)

// GetICDatasetLatest returns the latest version of a dataset
func GetICDatasetLatest(dataset *model.MynahICDataset) (*model.MynahICDatasetVersion, error) {
	latestVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions) - 1))
	if version, ok := dataset.Versions[latestVersionId]; ok {
		return version, nil
	}
	return nil, fmt.Errorf("dataset %s does not have a latest version", dataset.Uuid)
}

// GetICDatasetPrevious returns the previous version of a dataset
func GetICDatasetPrevious(dataset *model.MynahICDataset) (*model.MynahICDatasetVersion, error) {
	previousVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions) - 2))
	if version, ok := dataset.Versions[previousVersionId]; ok {
		return version, nil
	}
	return nil, fmt.Errorf("dataset %s does not have a previous version", dataset.Uuid)
}

// MakeICDatasetVersion creates a new ic dataset version.
// If a latest version exists, the files it references are versioned so that they won't be modified.
// Note: updates files in db, but does not update dataset
func MakeICDatasetVersion(dataset *model.MynahICDataset,
	user *model.MynahUser,
	storageProvider storage.StorageProvider,
	dbProvider db.DBProvider) (newVersion *model.MynahICDatasetVersion, err error) {

	newVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions)))

	//verify that the version id doesn't already exist
	if _, ok := dataset.Versions[newVersionId]; ok {
		return nil, fmt.Errorf("failed to create new dataset version, version id %s already exists", newVersionId)
	}

	//create a new version
	newVersion = &model.MynahICDatasetVersion{
		Files: make(map[model.MynahUuid]*model.MynahICDatasetFile),
		//report is nil for new versions
	}

	//add the new version to the mapping as well
	dataset.Versions[newVersionId] = newVersion

	//check if there is a previous version
	if latestVersion, err := GetICDatasetPrevious(dataset); err == nil {
		//map from fileid to the SHA1 version created
		newImageSHAVersions := make(map[model.MynahUuid]model.MynahFileVersionId)

		//version the files in the current "latest" so they won't be modified
		for fileId, fileData := range latestVersion.Files {
			//request the file
			if file, err := dbProvider.GetFile(fileId, user); err == nil {
				//generate a new sha1
				newId, err := storageProvider.GenerateSHA1Id(file)
				if err != nil {
					return nil, fmt.Errorf("failed to generate new version id for file: %s", err)
				}

				//only copy the file to the new version id if the id does not already exist
				if _, ok := file.Versions[newId]; !ok {
					//copy the latest version of the file to an explicit SHA1 version
					if err := storageProvider.CopyFile(file, model.LatestVersionId, newId); err != nil {
						return nil, fmt.Errorf("failed to version from current latest dataset version: %s", err)
					}

					//update the file in the database
					if err := dbProvider.UpdateFile(file, user, "versions"); err != nil {
						return nil, fmt.Errorf("failed to create new dataset version: %s", err)
					}
				}

				//set the new id (specific SHA1 version)
				fileData.ImageVersionId = newId

				//record for update in report section
				newImageSHAVersions[fileId] = newId

				//copy file data to the new version
				newVersion.Files[fileId] = model.CopyICDatasetFile(fileData)

			} else {
				return nil, fmt.Errorf("failed to version from current latest dataset version: %s", err)
			}
		}

		//update the report image tags to be SHA1 tags, not "latest"
		if latestVersion.Report != nil {
			for imageId, imageMetadata := range latestVersion.Report.ImageData {
				//set the specific image version
				if imageVersion, ok := newImageSHAVersions[imageId]; ok {
					imageMetadata.ImageVersionId = imageVersion
				} else {
					log.Warnf("dataset %s has image %s in report but not in dataset files, skipping version tag update", dataset.Uuid, imageId)
				}
			}
		}
	}

	//return ptr to new version
	return dataset.Versions[newVersionId], nil
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
