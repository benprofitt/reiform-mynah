// Copyright (c) 2022 by Reiform. All Rights Reserved.

package tools

import (
	"fmt"
	"reiform.com/mynah/containers"
	"reiform.com/mynah/db"
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
	"strconv"
)

// GetICDatasetPrevious returns the previous version of a dataset
// NOTE: versions must be present (not omitted)
func GetICDatasetPrevious(dataset *model.MynahICDataset) (*model.MynahICDatasetVersion, error) {
	previousVersionId := model.MynahDatasetVersionId(strconv.Itoa(len(dataset.Versions) - 2))
	if version, ok := dataset.Versions[previousVersionId]; ok {
		return version, nil
	}
	return nil, fmt.Errorf("dataset %s does not have a previous version", dataset.Uuid)
}

// ICDatasetVersionIterateNewToOld call function on dataset versions in order from latest to oldest
func ICDatasetVersionIterateNewToOld(dataset *model.MynahICDataset, handler func(version *model.MynahICDatasetVersion) (bool, error)) error {
	for i := len(dataset.Versions) - 1; i >= 0; i-- {
		versionId := model.MynahDatasetVersionId(strconv.Itoa(i))
		if version, ok := dataset.Versions[versionId]; ok {
			continueIteration, err := handler(version)
			if err != nil {
				return err
			}

			if !continueIteration {
				return nil
			}

		} else {
			return fmt.Errorf("malformed dataset version history with length %d, version %s does not exist",
				len(dataset.Versions), versionId)
		}
	}
	return nil
}

// FreezeICDatasetFileVersions freezes the ids of images in a dataset and report. Files are already locked.
func FreezeICDatasetFileVersions(version *model.MynahICDatasetVersion,
	user *model.MynahUser,
	storageProvider storage.StorageProvider,
	dbProvider db.DBProvider) error {
	//map from fileid to the SHA1 version created
	newImageSHAVersions := make(map[model.MynahUuid]model.MynahFileVersionId)

	fileIdSet := containers.NewUniqueSet[model.MynahUuid]()
	for fileId := range version.Files {
		fileIdSet.Union(fileId)
	}

	return dbProvider.Transaction(func(tx db.DBProvider) error {
		//batch request files
		files, err := tx.GetFiles(fileIdSet.Vals(), user)
		if err != nil {
			return err
		}

		//version the files in the current "latest" so they won't be modified
		for fileId, fileData := range version.Files {
			if file, ok := files[fileId]; ok {
				// get the latest version of the file
				latestVersion, err := file.GetFileVersion(model.LatestVersionId)
				if err != nil {
					return err
				}

				//generate a new sha1 based on the latest version
				newId, err := storageProvider.GenerateSHA1Id(file.Uuid, latestVersion)
				if err != nil {
					return fmt.Errorf("failed to generate new version id for file: %s", err)
				}

				//only copy the file to the new version id if the id does not already exist (idempotent)
				if _, ok := file.Versions[newId]; !ok {
					// create a new version
					file.Versions[newId] = model.NewMynahFileVersion(newId)
					//copy the latest version of the file to an explicit SHA1 version
					if err := storageProvider.CopyFile(file.Uuid, latestVersion, file.Versions[newId]); err != nil {
						return fmt.Errorf("failed to version file from current latest version: %s", err)
					}

					//update the file in the database with the new version
					if err := tx.UpdateFile(file, user, model.VersionsColName); err != nil {
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
	})
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
		newVersion.Mean = append([]float64(nil), latestVersion.Mean...)
		newVersion.StdDev = append([]float64(nil), latestVersion.StdDev...)

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
