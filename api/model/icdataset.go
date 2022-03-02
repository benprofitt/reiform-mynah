// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahICDataset Defines a dataset specifically for image classification
type MynahICDataset struct {
	//underlying mynah dataset
	MynahDataset `json:"dataset" xorm:"extends"`
	//all classes referenced in this dataset
	Classes []string `json:"classes" xorm:"TEXT 'classes'"`
}
