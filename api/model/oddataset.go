// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahODDatasetEntity defines an entity in the dataset
type MynahODDatasetEntity struct {
	//the current label assigned this entity
	CurrentLabel string `json:"current_label"`
	//the original label assigned to this entity
	OriginalLabel string `json:"original_label"`
	//the vertices for the bounding polygon
	Vertices [][]int `json:"vertices"`
}

// MynahODDatasetFile defines data about a file in the dataset
type MynahODDatasetFile struct {
	//the version of the file
	ImageVersionId MynahFileVersionId `json:"image_version_id"`
	//the entities in this file (map class -> entity ids)
	Entities map[string][]string `json:"entities"`
}

// MynahODDatasetVersion a version of this dataset
type MynahODDatasetVersion struct {
	//entities in the dataset by id
	Entities map[string]*MynahODDatasetEntity `json:"entities"`
	//files in the dataset
	Files map[string]*MynahODDatasetFile `json:"files"`
	//for each class, the files with that class
	FileEntities map[string][]string `json:"file_entities"`
}

// MynahODDataset Defines a dataset specifically for object classification
type MynahODDataset struct {
	//underlying mynah dataset
	MynahDataset `xorm:"extends"`
	//versions of this dataset
	Versions map[MynahDatasetVersionId]*MynahODDatasetVersion `json:"versions" xorm:"TEXT 'versions'"`
}

// NewODDataset creates a new dataset
func NewODDataset(creator *MynahUser) *MynahODDataset {
	return &MynahODDataset{
		MynahDataset: *NewDataset(creator),
		Versions:     make(map[MynahDatasetVersionId]*MynahODDatasetVersion),
	}
}
