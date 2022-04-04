// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahODProject Defines a project specifically for object classification
type MynahODProject struct {
	//underlying mynah project
	MynahProject `xorm:"extends"`
	//datasets that are part of this project: model.MynahOCDataset
	Datasets []string `json:"datasets" xorm:"TEXT 'datasets'"`
	//TODO reports
}
