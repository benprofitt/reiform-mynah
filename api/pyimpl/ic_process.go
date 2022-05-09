// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"fmt"
	"reiform.com/mynah/db"
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/tools"
)

// NewICProcessJobRequest creates a new ic process job request
func (p *localImplProvider) NewICProcessJobRequest(user *model.MynahUser,
	datasetId model.MynahUuid,
	currentVersion,
	prevVersion *model.MynahICDatasetVersion,
	tasks []model.MynahICProcessTaskType) (*ICProcessJobRequest, error) {

	var req ICProcessJobRequest

	//sanitize the task types
	for _, taskType := range tasks {
		if !model.ValidMynahICProcessTaskType(taskType) {
			return nil, fmt.Errorf("unknown ic process task type: %s", taskType)
		}
	}

	if prevVersion != nil {
		//add any previous task results (if exist)
		req.PreviousResults = prevVersion.TaskData
	} else {
		req.PreviousResults = make([]*model.MynahICProcessTaskData, 0)
	}

	//add tasks requested by user
	req.Tasks = make([]ICProcessJobRequestTask, len(tasks))
	//add the tasks
	for i := 0; i < len(tasks); i++ {
		req.Tasks[i].Type = tasks[i]
	}

	req.ConfigParams.ModelsPath = p.mynahSettings.StorageSettings.ModelsPath
	req.Dataset.DatasetUuid = datasetId
	req.Dataset.Mean = currentVersion.Mean
	req.Dataset.StdDev = currentVersion.StdDev
	req.Dataset.ClassFiles = make(map[string]map[string]ICProcessJobRequestFile)

	//accumulate the file uuids in this dataset
	fileUuidSet := tools.NewUniqueSet()
	for fileId := range currentVersion.Files {
		//add to set
		fileUuidSet.UuidsUnion(fileId)
	}

	//request the files in this dataset as a group
	files, err := p.dbProvider.GetFiles(fileUuidSet.UuidVals(), user)
	if err != nil {
		return nil, fmt.Errorf("failed to load dataset %s files for ic process job: %s", datasetId, err)
	}

	//record unique class names
	classNameSet := tools.NewUniqueSet()

	for fileId, classInfo := range currentVersion.Files {
		//include the class
		classNameSet.Union(classInfo.CurrentClass)

		var tmpPath string

		if f, ok := files[fileId]; ok {
			//get a path to this file, only use latest version id: other versions of file are immutable
			if path, err := p.storageProvider.GetTmpPath(f, model.LatestVersionId); err == nil {
				tmpPath = path
			} else {
				return nil, fmt.Errorf("failed to get temporary path to file %s: %s", fileId, err)
			}
		} else {
			return nil, fmt.Errorf("failed to get file %s: %s", fileId, err)
		}

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
				return nil, fmt.Errorf("unable to find file %s metadata using version id: %s", fileId, model.LatestVersionId)
			}
		} else {
			// (File may have been requested but this user may not have permission)
			return nil, fmt.Errorf("dataset file %s not found in files set for dataset (does user have permission?): %s", fileId, datasetId)
		}

		//create a mapping for this class if one doesn't exist
		if _, ok := req.Dataset.ClassFiles[classInfo.CurrentClass]; !ok {
			req.Dataset.ClassFiles[classInfo.CurrentClass] = make(map[string]ICProcessJobRequestFile)
		}

		//add the class -> tmp file -> data mapping
		req.Dataset.ClassFiles[classInfo.CurrentClass][tmpPath] = ICProcessJobRequestFile{
			Uuid:     fileId,
			Width:    width,
			Height:   height,
			Channels: channels,
			Mean:     classInfo.Mean,
			StdDev:   classInfo.StdDev,
		}
	}

	//set the classes
	req.Dataset.Classes = classNameSet.Vals()

	return &req, nil
}

// applyChanges modifies the dataset based on the result, and builds a report
func (d ICProcessJobResponse) applyChanges(dataset *model.MynahICDatasetVersion,
	report *model.MynahICDatasetReport,
	user *model.MynahUser,
	storageProvider storage.StorageProvider,
	dbProvider db.DBProvider) error {

	//set the dataset level mean and stddev
	dataset.Mean = d.Dataset.Mean
	dataset.StdDev = d.Dataset.StdDev

	for _, classFiles := range d.Dataset.ClassFiles {
		for _, fileData := range classFiles {
			//copy the new data into the dataset
			if datasetFile, ok := dataset.Files[fileData.Uuid]; ok {
				datasetFile.CurrentClass = fileData.CurrentClass
				datasetFile.Projections = fileData.Projections
				datasetFile.ConfidenceVectors = fileData.ConfidenceVectors
				datasetFile.Mean = fileData.Mean
				datasetFile.StdDev = fileData.StdDev
			} else {
				return fmt.Errorf("ic process returned fileid (%s) not tracked by dataset %s",
					fileData.Uuid, d.Dataset.DatasetUuid)
			}
		}
	}

	//apply task metadata to the dataset (may remove, modify files)
	for _, task := range d.Tasks {
		if err := task.Metadata.ApplyToDataset(dataset, task.Type); err != nil {
			return fmt.Errorf("failed to apply task of type %s result to dataset: %s", task.Type, err)
		}
	}

	//freeze the fileids so that the "latest" versions aren't modified by a different dataset
	if err := tools.FreezeICDatasetFileVersions(dataset, user, storageProvider, dbProvider); err != nil {
		return fmt.Errorf("failed to freeze file versions: %s", err)
	}

	//add remaining dataset files to the report (will have strict file versions)
	for fileId, fileData := range dataset.Files {
		//record this image for the report
		report.ImageIds = append(report.ImageIds, fileId)

		//add file data to report (will be updated by any completed tasks)
		report.ImageData[fileId] = &model.MynahICDatasetReportImageMetadata{
			ImageVersionId: fileData.ImageVersionId,
			Class:          fileData.CurrentClass,
			Point: model.MynahICDatasetReportPoint{
				X: 0,
				Y: 0,
			},
			OutlierTasks: []model.MynahICProcessTaskType{},
		}
	}

	//apply task metadata to the report
	for _, task := range d.Tasks {
		if err := task.Metadata.ApplyToReport(report, task.Type); err != nil {
			return fmt.Errorf("failed to apply task of type %s result to report: %s", task.Type, err)
		}
	}

	//compute breakdown
	for _, fileData := range report.ImageData {
		if _, ok := report.Breakdown[fileData.Class]; !ok {
			report.Breakdown[fileData.Class] = &model.MynahICDatasetReportBucket{
				Bad:        0,
				Acceptable: 0,
			}
		}

		//consider this file "bad" if it's an outlier in at least one task
		if len(fileData.OutlierTasks) > 0 {
			report.Breakdown[fileData.Class].Bad++
		} else {
			report.Breakdown[fileData.Class].Acceptable++
		}
	}
	return nil
}
