// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// ConfidenceVectors for a file
type ConfidenceVectors [][]float64

// MynahICDatasetFile file data
type MynahICDatasetFile struct {
	//the current clas
	CurrentClass string `json:"current_class"`
	//the original class
	OriginalClass string `json:"original_class"`
	//the confidence vectors
	ConfidenceVectors ConfidenceVectors `json:"confidence_vectors"`
	//projections
	Projections map[string][]int `json:"projections"`
}

// MynahICDataset Defines a dataset specifically for image classification
type MynahICDataset struct {
	//underlying mynah dataset
	MynahDataset `xorm:"extends"`
	//map of fileid -> file + class info
	Files map[string]*MynahICDatasetFile `json:"files" xorm:"TEXT 'files'"`
}
