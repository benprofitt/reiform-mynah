// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahICProcessTaskReportMetadata defines report metadata specific to some task
type MynahICProcessTaskReportMetadata interface {
}

// MynahICProcessTaskDiagnoseMislabeledImagesReportClass defines the files for a class
type MynahICProcessTaskDiagnoseMislabeledImagesReportClass struct {
	// the mislabeled files in the dataset
	Mislabeled []MynahUuid `json:"mislabeled"`
	// the correct files in the dataset
	Correct []MynahUuid `json:"correct"`
}

// MynahICProcessTaskDiagnoseMislabeledImagesReport records diagnosis report info for mislabeled images
type MynahICProcessTaskDiagnoseMislabeledImagesReport struct {
	ClassLabelErrors map[MynahClassName]*MynahICProcessTaskDiagnoseMislabeledImagesReportClass `json:"class_label_errors"`
}

// MynahICProcessTaskCorrectMislabeledImagesReportClass defines the files for a class
type MynahICProcessTaskCorrectMislabeledImagesReportClass struct {
	// the mislabeled files in the dataset that were corrected
	MislabeledCorrected []MynahUuid `json:"mislabeled_corrected"`
	// the mislabeled files in the dataset that were removed
	MislabeledRemoved []MynahUuid `json:"mislabeled_removed"`
	// the unchanged files in the dataset
	Unchanged []MynahUuid `json:"unchanged"`
}

// MynahICProcessTaskCorrectMislabeledImagesReport records correction
//report info for mislabeled images
type MynahICProcessTaskCorrectMislabeledImagesReport struct {
	ClassLabelErrors map[MynahClassName]*MynahICProcessTaskCorrectMislabeledImagesReportClass `json:"class_label_errors"`
}

// MynahICProcessTaskDiagnoseClassSplittingReportClass defines the class splitting predictions for a class
type MynahICProcessTaskDiagnoseClassSplittingReportClass struct {
	PredictedClassesCount int `json:"predicted_classes_count"`
}

// MynahICProcessTaskDiagnoseClassSplittingReport records diagnosis
//report info for class splitting
type MynahICProcessTaskDiagnoseClassSplittingReport struct {
	ClassesSplitting map[MynahClassName]*MynahICProcessTaskDiagnoseClassSplittingReportClass `json:"classes_splitting"`
}

// MynahICProcessTaskCorrectClassSplittingReportClass defines the new split classes
type MynahICProcessTaskCorrectClassSplittingReportClass struct {
	NewClasses []MynahClassName `json:"new_classes"`
}

// MynahICProcessTaskCorrectClassSplittingReport records correction
//report info for class splitting
type MynahICProcessTaskCorrectClassSplittingReport struct {
	ClassesSplitting map[MynahClassName]*MynahICProcessTaskCorrectClassSplittingReportClass `json:"classes_splitting"`
}

// MynahICProcessTaskReportData defines info about report task section
type MynahICProcessTaskReportData struct {
	//the type of the task
	Type MynahICProcessTaskType `json:"type"`
	//the task metadata for the report
	Metadata MynahICProcessTaskReportMetadata `json:"metadata"`
}

// MynahICDatasetReportPoint defines a plotted point and associated info
type MynahICDatasetReportPoint struct {
	// the id of the image this point plots
	FileId MynahUuid `json:"fileid"`
	//the version of the image
	ImageVersionId MynahFileVersionId `json:"image_version_id"`
	//the x,y coordinates
	X float64 `json:"x"`
	Y float64 `json:"y"`
	//the original class given
	OriginalClass MynahClassName `json:"original_class"`
}

// MynahICDatasetReport a mynah image classification report
type MynahICDatasetReport struct {
	//map from class to points (images) in class
	Points map[MynahClassName][]*MynahICDatasetReportPoint `json:"points"`
	//report data about tasks that were run
	Tasks []*MynahICProcessTaskReportData `json:"tasks"`
}

// NewICDatasetReport creates a new report
func NewICDatasetReport() *MynahICDatasetReport {
	return &MynahICDatasetReport{
		Points: make(map[MynahClassName][]*MynahICDatasetReportPoint),
		Tasks:  make([]*MynahICProcessTaskReportData, 0),
	}
}
