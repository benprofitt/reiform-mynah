// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
	"reiform.com/mynah/containers"
)

// ApplyChanges generates report for mislabeled images diagnosis
func (m MynahICProcessTaskDiagnoseMislabeledImagesMetadata) ApplyChanges(version *MynahICDatasetVersion, report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	classLabelErrors := make(map[MynahClassName]*MynahICProcessTaskDiagnoseMislabeledImagesReportClass)

	outliers := containers.NewUniqueSet[MynahUuid]()
	outliers.Union(m.Outliers...)

	for fileId, fileData := range version.Files {
		if _, ok := classLabelErrors[fileData.CurrentClass]; !ok {
			classLabelErrors[fileData.CurrentClass] = &MynahICProcessTaskDiagnoseMislabeledImagesReportClass{
				Mislabeled: make([]MynahUuid, 0),
				Correct:    make([]MynahUuid, 0),
			}
		}

		if outliers.Contains(fileId) {
			classLabelErrors[fileData.CurrentClass].Mislabeled = append(classLabelErrors[fileData.CurrentClass].Mislabeled, fileId)
		} else {
			classLabelErrors[fileData.CurrentClass].Correct = append(classLabelErrors[fileData.CurrentClass].Correct, fileId)
		}
	}

	//add to the report
	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type: taskType,
		Metadata: &MynahICProcessTaskDiagnoseMislabeledImagesReport{
			ClassLabelErrors: classLabelErrors,
		},
	})
	return nil
}

// ApplyChanges generates report for mislabeled images correction and updates dataset
func (m MynahICProcessTaskCorrectMislabeledImagesMetadata) ApplyChanges(version *MynahICDatasetVersion, report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	classLabelErrors := make(map[MynahClassName]*MynahICProcessTaskCorrectMislabeledImagesReportClass)

	removed := containers.NewUniqueSet[MynahUuid]()
	removed.Union(m.Removed...)
	corrected := containers.NewUniqueSet[MynahUuid]()
	corrected.Union(m.Corrected...)

	for fileId, fileData := range version.Files {
		if _, ok := classLabelErrors[fileData.CurrentClass]; !ok {
			classLabelErrors[fileData.CurrentClass] = &MynahICProcessTaskCorrectMislabeledImagesReportClass{
				MislabeledCorrected: make([]MynahUuid, 0),
				MislabeledRemoved:   make([]MynahUuid, 0),
				Unchanged:           make([]MynahUuid, 0),
			}
		}

		if removed.Contains(fileId) {
			classLabelErrors[fileData.CurrentClass].MislabeledRemoved = append(classLabelErrors[fileData.CurrentClass].MislabeledRemoved, fileId)
			// remove the file
			delete(version.Files, fileId)

		} else if corrected.Contains(fileId) {
			classLabelErrors[fileData.CurrentClass].MislabeledCorrected = append(classLabelErrors[fileData.CurrentClass].MislabeledCorrected, fileId)
		} else {
			classLabelErrors[fileData.CurrentClass].Unchanged = append(classLabelErrors[fileData.CurrentClass].Unchanged, fileId)
		}
	}

	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type: taskType,
		Metadata: &MynahICProcessTaskCorrectMislabeledImagesReport{
			ClassLabelErrors: classLabelErrors,
		},
	})
	return nil
}

// ApplyChanges generates report for class splitting diagnosis
func (m MynahICProcessTaskDiagnoseClassSplittingMetadata) ApplyChanges(version *MynahICDatasetVersion, report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	classesSplitting := make(map[MynahClassName]*MynahICProcessTaskDiagnoseClassSplittingReportClass)

	for class, classFiles := range m.PredictedClassSplits {
		classesSplitting[class] = &MynahICProcessTaskDiagnoseClassSplittingReportClass{
			PredictedClassesCount: len(classFiles),
		}
	}

	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type: taskType,
		Metadata: &MynahICProcessTaskDiagnoseClassSplittingReport{
			ClassesSplitting: classesSplitting,
		},
	})
	return nil
}

// ApplyChanges generates report for class splitting correction and updates dataset
func (m MynahICProcessTaskCorrectClassSplittingMetadata) ApplyChanges(version *MynahICDatasetVersion, report *MynahICDatasetReport, taskType MynahICProcessTaskType) error {
	classesSplitting := make(map[MynahClassName]*MynahICProcessTaskCorrectClassSplittingReportClass)

	for class, classSplit := range m.ActualClassSplits {
		newClasses := make([]MynahClassName, 0)
		for newClass := range classSplit {
			newClasses = append(newClasses, newClass)
		}

		classesSplitting[class] = &MynahICProcessTaskCorrectClassSplittingReportClass{
			NewClasses: newClasses,
		}
	}

	report.Tasks = append(report.Tasks, &MynahICProcessTaskReportData{
		Type: taskType,
		Metadata: &MynahICProcessTaskCorrectClassSplittingReport{
			ClassesSplitting: classesSplitting,
		},
	})
	return nil
}
