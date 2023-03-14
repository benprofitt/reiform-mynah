// Copyright (c) 2023 by Reiform. All Rights Reserved.

package report

import (
	"encoding/json"
	"errors"
	"fmt"
	"reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/models/types"
)

// MynahICProcessTaskReportMetadata defines report metadata specific to some task
type MynahICProcessTaskReportMetadata interface {
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

// MynahICDatasetVersionReportDataContents contains the type and task specific contents
type MynahICDatasetVersionReportDataContents struct {
	Type     dataset.MynahICProcessTaskType   `json:"type"`
	Metadata MynahICProcessTaskReportMetadata `json:"metadata"`
}

// MynahICDatasetVersionReportData defines info about report task section
type MynahICDatasetVersionReportData struct {
	ID       int64                                    `json:"-" xorm:"pk autoincr"`
	ReportId types.MynahUuid                          `json:"report_id" xorm:"varchar(36) not null index 'report_id'"`
	Contents *MynahICDatasetVersionReportDataContents `json:"contents" xorm:"TEXT 'contents'"`
}

func init() {
	db.RegisterTables(
		&MynahICDatasetVersionReportData{},
	)
}

// UnmarshalJSON deserializes the task metadata
func (t *MynahICDatasetVersionReportDataContents) UnmarshalJSON(bytes []byte) error {
	//check the task type
	var objMap map[string]*json.RawMessage

	if err := json.Unmarshal(bytes, &objMap); err != nil {
		return err
	}

	_, hasTypeField := objMap["type"]
	if !hasTypeField {
		return errors.New("ic process task missing 'type'")
	}

	if err := json.Unmarshal(*objMap["type"], &t.Type); err != nil {
		return fmt.Errorf("error deserializing ic process task: %s", err)
	}

	_, hasMetadataField := objMap["metadata"]
	if !hasMetadataField {
		return errors.New("ic process task missing 'metadata'")
	}

	var taskStruct interface{}

	switch t.Type {
	case dataset.ICProcessDiagnoseMislabeledImagesTask:
		taskStruct = &MynahICProcessTaskDiagnoseMislabeledImagesReport{}
	case dataset.ICProcessCorrectMislabeledImagesTask:
		taskStruct = &MynahICProcessTaskCorrectMislabeledImagesReport{}
	case dataset.ICProcessDiagnoseClassSplittingTask:
		taskStruct = &MynahICProcessTaskDiagnoseClassSplittingReport{}
	case dataset.ICProcessCorrectClassSplittingTask:
		taskStruct = &MynahICProcessTaskCorrectClassSplittingReport{}
	default:
		return fmt.Errorf("invalid ic processing type: %s", t.Type)
	}

	if err := json.Unmarshal(*objMap["metadata"], taskStruct); err != nil {
		return fmt.Errorf("error deserializing ic process task metadata (type: %s): %s", t.Type, err)
	}

	//set the task
	t.Metadata = taskStruct
	return nil
}
