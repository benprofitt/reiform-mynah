// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// ConfidenceVectors for a file
type ConfidenceVectors [][]float64

// MynahICProjectClassFileData file data in the project
type MynahICProjectClassFileData struct {
	//the current clas
	CurrentClass string `json:"current_class"`
	//the original class
	OriginalClass string `json:"original_class"`
	//the projections
	//TODO
	//the confidence vectors
	ConfidenceVectors ConfidenceVectors `json:"confidence_vectors"`
}

// MynahICProjectData defines project-level info about an ic dataset
type MynahICProjectData struct {
	//mapping from classes in the dataset to a mapping from file to file info
	Data map[string]map[string]MynahICProjectClassFileData `json:"data"`
}

// MynahICProject Defines a project specifically for image classification
type MynahICProject struct {
	//underlying mynah dataset
	MynahProject `json:"dataset" xorm:"extends"`
	//Datasets in this project (by uuid)
	DatasetAttributes map[string]MynahICProjectData `json:"dataset_attributes" xorm:"TEXT 'dataset_attributes'"`
}
