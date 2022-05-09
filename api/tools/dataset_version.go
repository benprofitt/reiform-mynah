// Copyright (c) 2022 by Reiform. All Rights Reserved.

package tools

import (
	"fmt"
	"reiform.com/mynah/db"
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
	"strconv"
)

// GetICDatasetLatest returns the latest version of a dataset
func GetICDatasetLatest(dataset *model.MynahICDataset) (*model.MynahICDatasetVersion, error) {
	if version, ok := dataset.Versions[dataset.LatestVersion]; ok {
		return version, nil
	}
	return nil, fmt.Errorf("dataset %s does not have a latest version", dataset.Uuid)
}

// GetICDatasetPrevious returns the previous version of a dataset
// NOTE: versions must be present (not omitted)
func GetICDatasetPrevious(dataset *model.MynahICDataset) (*model.MynahICDatasetVersion, error) {
	previousVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions) - 2))
	if version, ok := dataset.Versions[previousVersionId]; ok {
		return version, nil
	}
	return nil, fmt.Errorf("dataset %s does not have a previous version", dataset.Uuid)
}

// FreezeICDatasetFileVersions freezes the ids of images in a dataset and report
func FreezeICDatasetFileVersions(version *model.MynahICDatasetVersion,
	user *model.MynahUser,
	storageProvider storage.StorageProvider,
	dbProvider db.DBProvider) error {
	//map from fileid to the SHA1 version created
	newImageSHAVersions := make(map[model.MynahUuid]model.MynahFileVersionId)

	fileIdSet := NewUniqueSet()
	for fileId := range version.Files {
		fileIdSet.UuidsUnion(fileId)
	}
	//batch request files
	files, err := dbProvider.GetFiles(fileIdSet.UuidVals(), user)
	if err != nil {
		return err
	}

	//version the files in the current "latest" so they won't be modified
	for fileId, fileData := range version.Files {
		if file, ok := files[fileId]; ok {
			//generate a new sha1
			newId, err := storageProvider.GenerateSHA1Id(file)
			if err != nil {
				return fmt.Errorf("failed to generate new version id for file: %s", err)
			}

			//only copy the file to the new version id if the id does not already exist (idempotent)
			if _, ok := file.Versions[newId]; !ok {
				//copy the latest version of the file to an explicit SHA1 version
				if err := storageProvider.CopyFile(file, model.LatestVersionId, newId); err != nil {
					return fmt.Errorf("failed to version file from current latest version: %s", err)
				}

				//update the file in the database with the new version
				if err := dbProvider.UpdateFile(file, user, "versions"); err != nil {
					return fmt.Errorf("failed to update file while freezing file version: %s", err)
				}
			}

			//set the new id (specific SHA1 version)
			fileData.ImageVersionId = newId
			//record for update in report section
			newImageSHAVersions[fileId] = newId

		} else {
			return fmt.Errorf("failed to get file %s when freezing version", fileId)
		}
	}

	return nil
}

// MakeICDatasetVersion creates a new ic dataset version.
func MakeICDatasetVersion(dataset *model.MynahICDataset) (newVersion *model.MynahICDatasetVersion, err error) {
	newVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions)))

	//verify that the version id doesn't already exist
	if _, ok := dataset.Versions[newVersionId]; ok {
		return nil, fmt.Errorf("failed to create new dataset version, version id %s already exists", newVersionId)
	}

	//create a new version
	newVersion = model.NewICDatasetVersion()

	//add the new version to the mapping as well
	dataset.Versions[newVersionId] = newVersion

	//check if there is a previous version
	if latestVersion, err := GetICDatasetPrevious(dataset); err == nil {
		//copy the dataset level mean and stddev
		copy(newVersion.Mean, latestVersion.Mean)
		copy(newVersion.StdDev, latestVersion.StdDev)

		for fileId, fileData := range latestVersion.Files {
			//copy file data to the new version (applies 'latest' tag)
			newVersion.Files[fileId] = model.CopyICDatasetFile(fileData)
		}
	}
	dataset.LatestVersion = newVersionId
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
	//latestVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions) - 1))

	////check if there is a previous version
	//if _, ok := dataset.Versions[latestVersionId]; ok {
	//
	//	//TODO
	//}

	//create a new version
	dataset.Versions[newVersionId] = model.NewODDatasetVersion()

	//return ptr to new version
	return dataset.Versions[newVersionId], previousVersion, nil
}
