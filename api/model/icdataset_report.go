// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahICProcessTaskReportMetadata defines report metadata specific to some task
type MynahICProcessTaskReportMetadata interface {
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
	Class MynahClassName `json:"class"`
	//point for display in graph
	Point MynahICDatasetReportPoint `json:"point"`
	//whether this file was removed
	Removed bool `json:"removed"`
	//the tasks for which this image is an outlier
	OutlierTasks []MynahICProcessTaskType `json:"outlier_tasks"`
}

// MynahICProcessTaskReportData defines info about report task section
type MynahICProcessTaskReportData struct {
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
	Breakdown map[MynahClassName]*MynahICDatasetReportBucket `json:"breakdown"`
	//report data about tasks that were run
	Tasks []*MynahICProcessTaskReportData `json:"tasks"`
}

// NewICDatasetReport creates a new report
func NewICDatasetReport() *MynahICDatasetReport {
	return &MynahICDatasetReport{
		ImageIds:  make([]MynahUuid, 0),
		ImageData: make(map[MynahUuid]*MynahICDatasetReportImageMetadata),
		Breakdown: make(map[MynahClassName]*MynahICDatasetReportBucket),
		Tasks:     make([]*MynahICProcessTaskReportData, 0),
	}
}
