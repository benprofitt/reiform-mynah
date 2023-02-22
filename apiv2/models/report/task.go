// Copyright (c) 2023 by Reiform. All Rights Reserved.

package report

import (
	"reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/types"
)

// MynahICProcessTaskReportMetadata defines report metadata specific to some task
type MynahICProcessTaskReportMetadata interface {
}

// MynahICProcessTaskReportData defines info about report task section
type MynahICProcessTaskReportData struct {
	Type     dataset.MynahICProcessTaskType   `json:"type"`
	Metadata MynahICProcessTaskReportMetadata `json:"metadata"`
}

// MynahICProcessTaskDiagnoseMislabeledImagesReport records diagnosis report info for mislabeled images
type MynahICProcessTaskDiagnoseMislabeledImagesReport struct {
	ClassLabelErrors map[dataset.MynahClassName]struct {
		Mislabeled []types.MynahUuid `json:"mislabeled"`
		Correct    []types.MynahUuid `json:"correct"`
	} `json:"class_label_errors"`
}

// MynahICProcessTaskCorrectMislabeledImagesReport records correction
//report info for mislabeled images
type MynahICProcessTaskCorrectMislabeledImagesReport struct {
	ClassLabelErrors map[dataset.MynahClassName]struct {
		MislabeledCorrected []types.MynahUuid `json:"mislabeled_corrected"`
		MislabeledRemoved   []types.MynahUuid `json:"mislabeled_removed"`
		Unchanged           []types.MynahUuid `json:"unchanged"`
	} `json:"class_label_errors"`
}

// MynahICProcessTaskDiagnoseClassSplittingReport records diagnosis
//report info for class splitting
type MynahICProcessTaskDiagnoseClassSplittingReport struct {
	ClassesSplitting map[dataset.MynahClassName]struct {
		PredictedClassesCount int `json:"predicted_classes_count"`
	} `json:"classes_splitting"`
}

// MynahICProcessTaskCorrectClassSplittingReport records correction
//report info for class splitting
type MynahICProcessTaskCorrectClassSplittingReport struct {
	ClassesSplitting map[dataset.MynahClassName]struct {
		NewClasses []dataset.MynahClassName `json:"new_classes"`
	} `json:"classes_splitting"`
}
