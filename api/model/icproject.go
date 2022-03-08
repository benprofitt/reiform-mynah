// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahICProject Defines a project specifically for image classification
type MynahICProject struct {
	//underlying mynah dataset
	MynahProject `json:"dataset" xorm:"extends"`
	//datasets that are part of this project (by id)
	Datasets []string `json:"datasets" xorm:"TEXT 'datasets'"`
}
