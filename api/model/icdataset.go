// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

//Defines a dataset specifically for image classification
type MynahICDataset struct {
	//underlying mynah dataset
	MynahDataset `json:"dataset" xorm:"extends"`
}
