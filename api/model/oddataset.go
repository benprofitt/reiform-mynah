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
	//the entities in this file (map class -> entity ids)
	Entities map[string][]string `json:"entities"`
}

// MynahODDataset Defines a dataset specifically for object classification
type MynahODDataset struct {
	//underlying mynah dataset
	MynahDataset `xorm:"extends"`
	//entities in the dataset by id
	Entities map[string]*MynahODDatasetEntity `json:"entities" xorm:"TEXT 'entities'"`
	//files in the dataset
	Files map[string]*MynahODDatasetFile `json:"files" xorm:"TEXT 'files'"`
	//for each class, the files with that class
	FileEntities map[string][]string `json:"file_entities" xorm:"TEXT 'file_entities'"`
}
