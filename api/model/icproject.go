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

//defines information about a class within a dataset
type MynahICProjectClassData struct {
	//the files referenced in this dataset
	Files map[string]MynahICProjectClassFileData `json:"files"`
}

//defines project-level info about an ic dataset
type MynahICProjectData struct {
	//the ic dataset
	Dataset MynahICDataset `json:"dataset"`
	//mapping from classes in the dataset to file-level info
	Classes map[string]MynahICProjectClassData `json:"classes"`
}

//Defines a project specifically for image classification
type MynahICProject struct {
	//underlying mynah dataset
	MynahProject `json:"dataset" xorm:"extends"`
	//Datasets in this project
	Datasets []MynahICProjectData `json:"datasets" xorm:"TEXT 'datasets'"`
}
