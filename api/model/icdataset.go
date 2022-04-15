// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// ConfidenceVectors for a file
type ConfidenceVectors [][]float64

// MynahICDatasetFile file data
type MynahICDatasetFile struct {
	//the version id for the file
	ImageVersionId MynahFileVersionId `json:"image_version_id"`
	//the current clas
	CurrentClass string `json:"current_class"`
	//the original class
	OriginalClass string `json:"original_class"`
	//the confidence vectors
	ConfidenceVectors ConfidenceVectors `json:"confidence_vectors"`
	//projections
	Projections map[string][]int `json:"projections"`
}

// MynahICDatasetVersion defines a specific version of the dataset
type MynahICDatasetVersion struct {
	//map of fileid -> file + class info
	Files map[string]*MynahICDatasetFile `json:"files"`
	//reports generated for this dataset: model.MynahICDiagnosisReport
	Reports []string `json:"reports"`
}

// MynahICDataset Defines a dataset specifically for image classification
type MynahICDataset struct {
	//underlying mynah dataset
	MynahDataset `xorm:"extends"`
	//versions of the dataset
	Versions map[MynahDatasetVersionId]*MynahICDatasetVersion `json:"versions" xorm:"TEXT 'versions'"`
}

// NewICDataset creates a new dataset
func NewICDataset(creator *MynahUser) *MynahICDataset {
	return &MynahICDataset{
		MynahDataset: *NewDataset(creator),
		Versions:     make(map[MynahDatasetVersionId]*MynahICDatasetVersion),
	}
}
