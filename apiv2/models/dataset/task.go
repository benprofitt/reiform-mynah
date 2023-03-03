// Copyright (c) 2023 by Reiform. All Rights Reserved.

package dataset

import (
	"reiform.com/mynah-api/models/types"
)

// MynahICProcessTaskData defines the result of running a task on an ic dataset
type MynahICProcessTaskData struct {
	Type     MynahICProcessTaskType     `json:"type"`
	Metadata MynahICProcessTaskMetadata `json:"metadata"`
}

// MynahICProcessTaskMetadata defines metadata specific to some task
type MynahICProcessTaskMetadata interface {
}

// MynahICProcessTaskDiagnoseMislabeledImagesMetadata is metadata for
//diagnosing mislabeled images task response
type MynahICProcessTaskDiagnoseMislabeledImagesMetadata struct {
	Outliers []types.MynahUuid `json:"outliers"`
}

// MynahICProcessTaskCorrectMislabeledImagesMetadata is metadata for
//correcting mislabeled images task response
type MynahICProcessTaskCorrectMislabeledImagesMetadata struct {
	Removed   []types.MynahUuid `json:"removed"`
	Corrected []types.MynahUuid `json:"corrected"`
}

// MynahICProcessTaskDiagnoseClassSplittingMetadata is metadata for
//diagnosing class splitting task response
type MynahICProcessTaskDiagnoseClassSplittingMetadata struct {
	PredictedClassSplits map[MynahClassName][][]types.MynahUuid `json:"predicted_class_splits"`
}

// MynahICProcessTaskCorrectClassSplittingMetadata is metadata for
//correcting class splitting task response
type MynahICProcessTaskCorrectClassSplittingMetadata struct {
	ActualClassSplits map[MynahClassName]map[MynahClassName][]types.MynahUuid `json:"actual_class_splits"`
}
