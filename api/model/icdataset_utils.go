// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
	"errors"
	"fmt"
)

// ApplyToDataset task changes for mislabeled images diagnosis
func (m MynahICProcessTaskDiagnoseMislabeledImagesMetadata) ApplyToDataset(version *MynahICDatasetVersion, taskType MynahICProcessTaskType) error {
	//TODO
	return nil
}

// ApplyToReport generates report for mislabeled images diagnosis
func (m MynahICProcessTaskDiagnoseMislabeledImagesMetadata) ApplyToReport(report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	for _, fileId := range m.Outliers {
		//get the file data in the report
		if fileData, ok := report.ImageData[fileId]; ok {
			//reference this task
			fileData.OutlierTasks = append(fileData.OutlierTasks, taskType)
		} else {
			return fmt.Errorf("%s task referenced outlier fileid which is not in dataset: %s", taskType, fileId)
		}
	}

	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type:     taskType,
		Metadata: &MynahICProcessTaskDiagnoseMislabeledImagesReport{},
	})
	return nil
}

// ApplyToDataset task changes for mislabeled images correction
func (m MynahICProcessTaskCorrectMislabeledImagesMetadata) ApplyToDataset(version *MynahICDatasetVersion, taskType MynahICProcessTaskType) error {
	//TODO
	return errors.New("MynahICProcessTaskCorrectMislabeledImagesMetadata dataset task changes undefined")
}

// ApplyToReport generates report for mislabeled images correction
func (m MynahICProcessTaskCorrectMislabeledImagesMetadata) ApplyToReport(report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type:     taskType,
		Metadata: &MynahICProcessTaskCorrectMislabeledImagesReport{},
	})
	return nil
}

// ApplyToDataset task changes for class splitting diagnosis
func (m MynahICProcessTaskDiagnoseClassSplittingMetadata) ApplyToDataset(version *MynahICDatasetVersion, taskType MynahICProcessTaskType) error {
	//TODO
	return errors.New("MynahICProcessTaskDiagnoseClassSplittingMetadata dataset task changes undefined")
}

// ApplyToReport generates report for class splitting diagnosis
func (m MynahICProcessTaskDiagnoseClassSplittingMetadata) ApplyToReport(report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type:     taskType,
		Metadata: &MynahICProcessTaskDiagnoseClassSplittingReport{},
	})
	return nil
}

// ApplyToDataset task changes for class splitting correction
func (m MynahICProcessTaskCorrectClassSplittingMetadata) ApplyToDataset(version *MynahICDatasetVersion, taskType MynahICProcessTaskType) error {
	//TODO
	return errors.New("MynahICProcessTaskCorrectClassSplittingMetadata dataset task changes undefined")
}

// ApplyToReport generates report for class splitting correction
func (m MynahICProcessTaskCorrectClassSplittingMetadata) ApplyToReport(report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type:     taskType,
		Metadata: &MynahICProcessTaskCorrectClassSplittingReport{},
	})
	return nil
}

// ApplyToDataset task changes for lighting conditions diagnosis
func (m MynahICProcessTaskDiagnoseLightingConditionsMetadata) ApplyToDataset(version *MynahICDatasetVersion, taskType MynahICProcessTaskType) error {
	//TODO
	return errors.New("MynahICProcessTaskDiagnoseLightingConditionsMetadata dataset task changes undefined")
}

// ApplyToReport generates report for lighting conditions diagnosis
func (m MynahICProcessTaskDiagnoseLightingConditionsMetadata) ApplyToReport(report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type:     taskType,
		Metadata: &MynahICProcessTaskDiagnoseLightingConditionsReport{},
	})
	return nil
}

// ApplyToDataset task changes for lighting conditions correction
func (m MynahICProcessTaskCorrectLightingConditionsMetadata) ApplyToDataset(version *MynahICDatasetVersion,
	taskType MynahICProcessTaskType) error {
	//for fileId := range m.Removed {
	//
	//}
	//
	//for fileId := range m.Corrected {
	//
	//}
	return nil
}

// ApplyToReport generates report for lighting conditions correction
func (m MynahICProcessTaskCorrectLightingConditionsMetadata) ApplyToReport(report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type:     taskType,
		Metadata: &MynahICProcessTaskCorrectLightingConditionsReport{},
	})
	return nil
}

// ApplyToDataset task changes for image blur diagnosis
func (m MynahICProcessTaskDiagnoseImageBlurMetadata) ApplyToDataset(version *MynahICDatasetVersion, taskType MynahICProcessTaskType) error {
	//TODO

	return errors.New("MynahICProcessTaskDiagnoseImageBlurMetadata dataset task changes undefined")
}

// ApplyToReport generates report for image blur diagnosis
func (m MynahICProcessTaskDiagnoseImageBlurMetadata) ApplyToReport(report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type:     taskType,
		Metadata: &MynahICProcessTaskDiagnoseImageBlurReport{},
	})
	return nil
}

// ApplyToDataset task changes for image blur correction
func (m MynahICProcessTaskCorrectImageBlurMetadata) ApplyToDataset(version *MynahICDatasetVersion, taskType MynahICProcessTaskType) error {
	//TODO

	return errors.New("MynahICProcessTaskCorrectImageBlurMetadata dataset task changes undefined")
}

// ApplyToReport generates report for image blur correction
func (m MynahICProcessTaskCorrectImageBlurMetadata) ApplyToReport(report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type:     taskType,
		Metadata: &MynahICProcessTaskCorrectImageBlurReport{},
	})
	return nil
}
