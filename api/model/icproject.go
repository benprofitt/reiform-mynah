// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahICProject Defines a project specifically for image classification
type MynahICProject struct {
	//underlying mynah project
	MynahProject `xorm:"extends"`
	//datasets that are part of this project: model.MynahICDataset
	Datasets []string `json:"datasets" xorm:"TEXT 'datasets'"`
	//reports generated for this project: model.MynahICDiagnosisReport
	Reports []string `json:"reports" xorm:"TEXT 'reports'"`
}
