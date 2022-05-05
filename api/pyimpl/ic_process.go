// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"encoding/json"
	"errors"
	"fmt"
	"reiform.com/mynah/model"
	"reiform.com/mynah/tools"
)

// create new task data structs by type identifier
var taskMetadataConstructor = map[model.MynahICProcessTaskType]func() ICDatasetTransformer{
	model.ICProcessDiagnoseMislabeledImagesTask: func() ICDatasetTransformer {
		return &ICProcessJobResponseTaskDiagnoseMislabeledImagesMetadata{}
	},
	model.ICProcessCorrectMislabeledImagesTask: func() ICDatasetTransformer {
		return &ICProcessJobResponseTaskCorrectMislabeledImagesMetadata{}
	},
	model.ICProcessDiagnoseClassSplittingTask: func() ICDatasetTransformer {
		return &ICProcessJobResponseTaskDiagnoseClassSplittingMetadata{}
	},
	model.ICProcessCorrectClassSplittingTask: func() ICDatasetTransformer {
		return &ICProcessJobResponseTaskCorrectClassSplittingMetadata{}
	},
	model.ICProcessDiagnoseLightingConditionsTask: func() ICDatasetTransformer {
		return &ICProcessJobResponseTaskDiagnoseLightingConditionsMetadata{}
	},
	model.ICProcessCorrectLightingConditionsTask: func() ICDatasetTransformer {
		return &ICProcessJobResponseTaskCorrectLightingConditionsMetadata{}
	},
	model.ICProcessDiagnoseImageBlurTask: func() ICDatasetTransformer {
		return &ICProcessJobResponseTaskDiagnoseImageBlurMetadata{}
	},
	model.ICProcessCorrectImageBlurTask: func() ICDatasetTransformer {
		return &ICProcessJobResponseTaskCorrectImageBlurMetadata{}
	},
}

// UnmarshalJSON deserializes the task metadata
func (t *ICProcessJobResponseTask) UnmarshalJSON(bytes []byte) error {
	//check the task type
	var objMap map[string]*json.RawMessage

	if err := json.Unmarshal(bytes, &objMap); err != nil {
		return err
	}

	_, hasTypeField := objMap["type"]
	_, hasMetadataField := objMap["metadata"]

	if hasTypeField && hasMetadataField {
		//deserialize the task id
		if err := json.Unmarshal(*objMap["type"], &t.Type); err != nil {
			return fmt.Errorf("error deserializing ic process result task: %s", err)
		}

		//look for the task struct type
		if taskStructFn, ok := taskMetadataConstructor[t.Type]; ok {
			taskStruct := taskStructFn()

			//unmarshal the actual task contents
			if err := json.Unmarshal(*objMap["metadata"], taskStruct); err != nil {
				return fmt.Errorf("error deserializing ic process task metadata (type: %s): %s", t.Type, err)
			}

			//set the task
			t.Metadata = taskStruct
			return nil

		} else {
			return fmt.Errorf("unknown ic process task type: %s", t.Type)
		}

	} else {
		return errors.New("ic process task missing 'type' or 'metadata'")
	}
}

// NewICProcessJobRequest creates a new ic process job request
func (p *localImplProvider) NewICProcessJobRequest(user *model.MynahUser, datasetId model.MynahUuid, dataset *model.MynahICDatasetVersion, tasks []model.MynahICProcessTaskType) (*ICProcessJobRequest, error) {
	var req ICProcessJobRequest

	//sanitize the task types
	for _, taskType := range tasks {
		if _, ok := taskMetadataConstructor[taskType]; !ok {
			return nil, fmt.Errorf("unknown ic process task type: %s", taskType)
		}
	}

	req.Tasks = make([]ICProcessJobRequestTask, len(tasks))
	//add the tasks
	for i := 0; i < len(tasks); i++ {
		req.Tasks[i].Type = tasks[i]
	}

	req.ConfigParams.ModelsPath = p.mynahSettings.StorageSettings.ModelsPath
	req.Dataset.DatasetUuid = datasetId
	req.Dataset.Mean = dataset.Mean
	req.Dataset.StdDev = dataset.StdDev
	req.Dataset.ClassFiles = make(map[string]map[string]ICProcessJobRequestFile)

	//accumulate the file uuids in this dataset
	fileUuidSet := tools.NewUniqueSet()
	for fileId := range dataset.Files {
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

	for fileId, classInfo := range dataset.Files {
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

// apply task changes for mislabeled images diagnosis
func (m ICProcessJobResponseTaskDiagnoseMislabeledImagesMetadata) apply(version *model.MynahICDatasetVersion, taskType model.MynahICProcessTaskType) error {
	for _, fileId := range m.Outliers {
		//get the file data
		if fileData, ok := version.Report.ImageData[fileId]; ok {
			//reference this task
			fileData.OutlierTasks = append(fileData.OutlierTasks, taskType)
		} else {
			return fmt.Errorf("%s task referenced outlier fileid which is not in dataset: %s", taskType, fileId)
		}
	}
	//add to the report
	version.Report.Tasks = append(version.Report.Tasks, model.MynahICProcessTaskReport{
		Type:     taskType,
		Metadata: &model.MynahICProcessTaskDiagnoseMislabeledImagesReport{},
	})
	return nil
}

// apply task changes for mislabeled images correction
func (m ICProcessJobResponseTaskCorrectMislabeledImagesMetadata) apply(version *model.MynahICDatasetVersion, taskType model.MynahICProcessTaskType) error {
	//TODO
	//add to the report
	version.Report.Tasks = append(version.Report.Tasks, model.MynahICProcessTaskReport{
		Type:     taskType,
		Metadata: &model.MynahICProcessTaskCorrectMislabeledImagesReport{},
	})
	return errors.New("ICProcessJobResponseTaskCorrectMislabeledImagesMetadata dataset task changes undefined")
}

// apply task changes for class splitting diagnosis
func (m ICProcessJobResponseTaskDiagnoseClassSplittingMetadata) apply(version *model.MynahICDatasetVersion, taskType model.MynahICProcessTaskType) error {
	//TODO
	//add to the report
	version.Report.Tasks = append(version.Report.Tasks, model.MynahICProcessTaskReport{
		Type:     taskType,
		Metadata: &model.MynahICProcessTaskDiagnoseClassSplittingReport{},
	})
	return errors.New("ICProcessJobResponseTaskDiagnoseClassSplittingMetadata dataset task changes undefined")
}

// apply task changes for class splitting  correction
func (m ICProcessJobResponseTaskCorrectClassSplittingMetadata) apply(version *model.MynahICDatasetVersion, taskType model.MynahICProcessTaskType) error {
	//TODO
	version.Report.Tasks = append(version.Report.Tasks, model.MynahICProcessTaskReport{
		Type:     taskType,
		Metadata: &model.MynahICProcessTaskCorrectClassSplittingReport{},
	})
	return errors.New("ICProcessJobResponseTaskCorrectClassSplittingMetadata dataset task changes undefined")
}

// apply task changes for lighting conditions diagnosis
func (m ICProcessJobResponseTaskDiagnoseLightingConditionsMetadata) apply(version *model.MynahICDatasetVersion, taskType model.MynahICProcessTaskType) error {
	//TODO
	version.Report.Tasks = append(version.Report.Tasks, model.MynahICProcessTaskReport{
		Type:     taskType,
		Metadata: &model.MynahICProcessTaskDiagnoseLightingConditionsReport{},
	})
	return errors.New("ICProcessJobResponseTaskDiagnoseLightingConditionsMetadata dataset task changes undefined")
}

// apply task changes for lighting conditions correction
func (m ICProcessJobResponseTaskCorrectLightingConditionsMetadata) apply(version *model.MynahICDatasetVersion, taskType model.MynahICProcessTaskType) error {
	//for fileId := range m.Removed {
	//
	//}
	//
	//for fileId := range m.Corrected {
	//
	//}
	version.Report.Tasks = append(version.Report.Tasks, model.MynahICProcessTaskReport{
		Type:     taskType,
		Metadata: &model.MynahICProcessTaskCorrectLightingConditionsReport{},
	})
	return nil
}

// apply task changes for image blur diagnosis
func (m ICProcessJobResponseTaskDiagnoseImageBlurMetadata) apply(version *model.MynahICDatasetVersion, taskType model.MynahICProcessTaskType) error {
	//TODO
	version.Report.Tasks = append(version.Report.Tasks, model.MynahICProcessTaskReport{
		Type:     taskType,
		Metadata: &model.MynahICProcessTaskDiagnoseImageBlurReport{},
	})
	return errors.New("ICProcessJobResponseTaskDiagnoseImageBlurMetadata dataset task changes undefined")
}

// apply task changes for image blur correction
func (m ICProcessJobResponseTaskCorrectImageBlurMetadata) apply(version *model.MynahICDatasetVersion, taskType model.MynahICProcessTaskType) error {
	//TODO
	version.Report.Tasks = append(version.Report.Tasks, model.MynahICProcessTaskReport{
		Type:     taskType,
		Metadata: &model.MynahICProcessTaskCorrectImageBlurReport{},
	})
	return errors.New("ICProcessJobResponseTaskCorrectImageBlurMetadata dataset task changes undefined")
}

// apply modifies the dataset based on the ic process result
func (d ICProcessJobResponse) apply(dataset *model.MynahICDatasetVersion) error {
	//set the dataset level mean and stddev
	dataset.Mean = d.Dataset.Mean
	dataset.StdDev = d.Dataset.StdDev

	//create a report for this ic process
	dataset.Report = model.NewMynahICDatasetReport()

	for _, classFiles := range d.Dataset.ClassFiles {
		for _, fileData := range classFiles {
			//record this image for the report
			dataset.Report.ImageIds = append(dataset.Report.ImageIds, fileData.Uuid)

			//copy the new data into the dataset
			if datasetFile, ok := dataset.Files[fileData.Uuid]; ok {
				datasetFile.CurrentClass = fileData.CurrentClass
				datasetFile.Projections = fileData.Projections
				datasetFile.ConfidenceVectors = fileData.ConfidenceVectors
				datasetFile.Mean = fileData.Mean
				datasetFile.StdDev = fileData.StdDev

				//add file data to report (will be updated by any completed tasks)
				dataset.Report.ImageData[fileData.Uuid] = &model.MynahICDatasetReportImageMetadata{
					//this dataset version should be frozen to avoid other processes modifying the latest version of files
					//(see tools/dataset_version.go)
					ImageVersionId: model.LatestVersionId,
					Class:          fileData.CurrentClass,
					Point: model.MynahICDatasetReportPoint{
						X: 0,
						Y: 0,
					},
					OutlierTasks: []model.MynahICProcessTaskType{},
				}

			} else {
				return fmt.Errorf("ic process returned fileid (%s) not tracked by dataset %s",
					fileData.Uuid, d.Dataset.DatasetUuid)
			}
		}
	}

	//apply task changes against the updated dataset and the new report
	for _, task := range d.Tasks {
		if err := task.Metadata.apply(dataset, task.Type); err != nil {
			return fmt.Errorf("failed to apply task of type %s result to dataset: %s", task.Type, err)
		}
	}

	//update breakdown
	for _, fileData := range dataset.Report.ImageData {
		//add the class to the breakdown if not already added
		if _, ok := dataset.Report.Breakdown[fileData.Class]; !ok {
			dataset.Report.Breakdown[fileData.Class] = &model.MynahICDatasetReportBucket{
				Bad:        0,
				Acceptable: 0,
			}
		}

		//consider this file "bad" if it's an outlier in at least one task
		if len(fileData.OutlierTasks) > 0 {
			dataset.Report.Breakdown[fileData.Class].Bad++
		} else {
			dataset.Report.Breakdown[fileData.Class].Acceptable++
		}
	}

	return nil
}
