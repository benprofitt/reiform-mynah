// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ConfidenceVectors for a file
type ConfidenceVectors [][]float64

// MynahICProcessTaskType is a task type name
type MynahICProcessTaskType string

const (
	ICProcessDiagnoseMislabeledImagesTask   MynahICProcessTaskType = "ic::diagnose::mislabeled_images"
	ICProcessCorrectMislabeledImagesTask    MynahICProcessTaskType = "ic::correct::mislabeled_images"
	ICProcessDiagnoseClassSplittingTask     MynahICProcessTaskType = "ic::diagnose::class_splitting"
	ICProcessCorrectClassSplittingTask      MynahICProcessTaskType = "ic::correct::class_splitting"
	ICProcessDiagnoseLightingConditionsTask MynahICProcessTaskType = "ic::diagnose::lighting_conditions"
	ICProcessCorrectLightingConditionsTask  MynahICProcessTaskType = "ic::correct::lighting_conditions"
	ICProcessDiagnoseImageBlurTask          MynahICProcessTaskType = "ic::diagnose::image_blur"
	ICProcessCorrectImageBlurTask           MynahICProcessTaskType = "ic::correct::image_blur"
)

// MynahICProcessTaskReportMetadata defines report metadata specific to some task
type MynahICProcessTaskReportMetadata interface {
}

// MynahICDatasetReportPoint a cartesian point
type MynahICDatasetReportPoint struct {
	X int64 `json:"x"`
	Y int64 `json:"y"`
}

// MynahICDatasetReportBucket classifications for images in class
type MynahICDatasetReportBucket struct {
	Bad        int `json:"bad"`
	Acceptable int `json:"acceptable"`
}

// MynahICDatasetReportImageMetadata defines the metadata associated with an image
type MynahICDatasetReportImageMetadata struct {
	//the version id for the file
	ImageVersionId MynahFileVersionId `json:"image_version_id"`
	//the class
	Class string `json:"class"`
	//point for display in graph
	Point MynahICDatasetReportPoint `json:"point"`
	//the tasks for which this image is an outlier
	OutlierTasks []MynahICProcessTaskType `json:"outlier_tasks"`
}

// MynahICProcessTaskDiagnoseMislabeledImagesReport records diagnosis report info for mislabeled images
type MynahICProcessTaskDiagnoseMislabeledImagesReport struct {
}

// MynahICProcessTaskCorrectMislabeledImagesReport records correction
//report info for mislabeled images
type MynahICProcessTaskCorrectMislabeledImagesReport struct {
}

// MynahICProcessTaskDiagnoseClassSplittingReport records diagnosis
//report info for class splitting
type MynahICProcessTaskDiagnoseClassSplittingReport struct {
}

// MynahICProcessTaskCorrectClassSplittingReport records correction
//report info for class splitting
type MynahICProcessTaskCorrectClassSplittingReport struct {
}

// MynahICProcessTaskDiagnoseLightingConditionsReport records diagnosis
//report info for lighting conditions
type MynahICProcessTaskDiagnoseLightingConditionsReport struct {
}

// MynahICProcessTaskCorrectLightingConditionsReport records correction
//report info for lighting conditions
type MynahICProcessTaskCorrectLightingConditionsReport struct {
}

// MynahICProcessTaskDiagnoseImageBlurReport records diagnosis
//report info for blur
type MynahICProcessTaskDiagnoseImageBlurReport struct {
}

// MynahICProcessTaskCorrectImageBlurReport records correction
//report info for blur
type MynahICProcessTaskCorrectImageBlurReport struct {
}

// MynahICProcessTaskReport defines info about report task section
type MynahICProcessTaskReport struct {
	//the type of the task
	Type MynahICProcessTaskType `json:"type"`
	//the task metadata for the report
	Metadata MynahICProcessTaskReportMetadata `json:"metadata"`
}

// MynahICDatasetReport a mynah image classification report
type MynahICDatasetReport struct {
	//all of the images, in the order to display
	ImageIds []MynahUuid `json:"image_ids"`
	//the images included in this report, map fileid to metadata
	ImageData map[MynahUuid]*MynahICDatasetReportImageMetadata `json:"image_data"`
	//the class breakdown table info, map class to buckets
	Breakdown map[string]*MynahICDatasetReportBucket `json:"breakdown"`
	//data about tasks that were run
	Tasks []MynahICProcessTaskReport `json:"tasks"`
}

// MynahICDatasetFile file data
type MynahICDatasetFile struct {
	//the version id for the file
	ImageVersionId MynahFileVersionId `json:"image_version_id"`
	//the current clas
	CurrentClass string `json:"current_class"`
	//the original class
	OriginalClass string `json:"original_class"`
	//the confidence vectors
	ConfidenceVectors ConfidenceVectors `json:"confidence_vectors"`
	//projections
	Projections map[string][]int `json:"projections"`
	//the mean of the this file's channels
	Mean []float64 `json:"mean"`
	//the stddev of this file's channels
	StdDev []float64 `json:"std_dev"`
}

// MynahICDatasetVersion defines a specific version of the dataset
type MynahICDatasetVersion struct {
	//map of fileid -> file + class info
	Files map[MynahUuid]*MynahICDatasetFile `json:"files"`
	//the mean of the channels of images in the dataset
	Mean []float64 `json:"mean"`
	//the stddev of the channels of images in the dataset
	StdDev []float64 `json:"std_dev"`
	//the unified report
	Report *MynahICDatasetReport `json:"report"`
}

// MynahICDataset Defines a dataset specifically for image classification
type MynahICDataset struct {
	//underlying mynah dataset
	MynahDataset `xorm:"extends"`
	//versions of the dataset
	Versions map[MynahDatasetVersionId]*MynahICDatasetVersion `json:"versions" xorm:"TEXT 'versions'"`
}

// NewICDataset creates a new dataset
func NewICDataset(creator *MynahUser) *MynahICDataset {
	return &MynahICDataset{
		MynahDataset: *NewDataset(creator),
		Versions:     make(map[MynahDatasetVersionId]*MynahICDatasetVersion),
	}
}

// NewICDatasetFile creates a new dataset file record
func NewICDatasetFile() *MynahICDatasetFile {
	return &MynahICDatasetFile{
		ImageVersionId:    LatestVersionId,
		CurrentClass:      "",
		OriginalClass:     "",
		ConfidenceVectors: make(ConfidenceVectors, 0),
		Projections:       make(map[string][]int),
		Mean:              make([]float64, 0),
		StdDev:            make([]float64, 0),
	}
}

// NewMynahICDatasetReport creates a new report
func NewMynahICDatasetReport() *MynahICDatasetReport {
	return &MynahICDatasetReport{
		ImageIds:  make([]MynahUuid, 0),
		ImageData: make(map[MynahUuid]*MynahICDatasetReportImageMetadata),
		Breakdown: make(map[string]*MynahICDatasetReportBucket),
		Tasks:     make([]MynahICProcessTaskReport, 0),
	}
}

// CopyICDatasetFile creates a new dataset file record by copying another
func CopyICDatasetFile(other *MynahICDatasetFile) *MynahICDatasetFile {
	var confidenceVectors ConfidenceVectors
	copy(confidenceVectors, other.ConfidenceVectors)

	projections := make(map[string][]int)
	for key, value := range other.Projections {
		projections[key] = value
	}

	var mean []float64
	copy(mean, other.Mean)

	var stdDev []float64
	copy(stdDev, other.StdDev)

	return &MynahICDatasetFile{
		ImageVersionId:    LatestVersionId,
		CurrentClass:      other.CurrentClass,
		OriginalClass:     other.OriginalClass,
		ConfidenceVectors: confidenceVectors,
		Projections:       projections,
		Mean:              mean,
		StdDev:            stdDev,
	}
}

// map from task type to report structure
var mynahICProcessTaskReportConstructor = map[MynahICProcessTaskType]func() MynahICProcessTaskReportMetadata{
	ICProcessDiagnoseMislabeledImagesTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskDiagnoseMislabeledImagesReport{}
	},
	ICProcessCorrectMislabeledImagesTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskCorrectMislabeledImagesReport{}
	},
	ICProcessDiagnoseClassSplittingTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskDiagnoseClassSplittingReport{}
	},
	ICProcessCorrectClassSplittingTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskCorrectClassSplittingReport{}
	},
	ICProcessDiagnoseLightingConditionsTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskDiagnoseLightingConditionsReport{}
	},
	ICProcessCorrectLightingConditionsTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskCorrectLightingConditionsReport{}
	},
	ICProcessDiagnoseImageBlurTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskDiagnoseImageBlurReport{}
	},
	ICProcessCorrectImageBlurTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskCorrectImageBlurReport{}
	},
}

// UnmarshalJSON deserializes the report metadata
func (t *MynahICProcessTaskReport) UnmarshalJSON(bytes []byte) error {
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
			return fmt.Errorf("error deserializing ic process report task: %s", err)
		}

		//create the task struct
		if taskStructFn, ok := mynahICProcessTaskReportConstructor[t.Type]; ok {
			taskStruct := taskStructFn()
			//unmarshal the actual task contents
			if err := json.Unmarshal(*objMap["metadata"], taskStruct); err != nil {
				return fmt.Errorf("error deserializing ic process task report (type: %s): %s", t.Type, err)
			}

			//set the task
			t.Metadata = taskStruct
			return nil
		} else {
			return fmt.Errorf("unknown ic process task type: %s", t.Type)
		}

	} else {
		return errors.New("ic process task report missing 'type' or 'metadata'")
	}
}
