// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

type MynahICProjectClassFileData struct {
	//the current clas
	CurrentClass string `json:"current_class"`
	//the original class
	OriginalClass string `json:"original_class"`
	//the projections
	//TODO
	//the confidence vectors
	ConfidenceVectors [][]float64 `json:"confidence_vectors"`
}

//defines project-level info about an ic dataset
type MynahICProjectData struct {
	//mapping from classes in the dataset to a mapping from file to file info
	Data map[string]map[string]MynahICProjectClassFileData `json:"data"`
}

//Defines a project specifically for image classification
type MynahICProject struct {
	//underlying mynah dataset
	MynahProject `json:"dataset" xorm:"extends"`
	//Datasets in this project (by uuid)
	DatasetAttributes map[string]MynahICProjectData `json:"dataset_attributes" xorm:"TEXT 'dataset_attributes'"`
}
