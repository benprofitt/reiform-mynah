// Copyright (c) 2023 by Reiform. All Rights Reserved.

package models

import (
	"fmt"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/types"
)

type MynahDatasetType string

const (
	ImageClassificationDataset MynahDatasetType = "image_classification"
)

// MynahDataset defines a mynah dataset
type MynahDataset struct {
	DatasetId    types.MynahUuid  `json:"dataset_id" xorm:"varchar(36) pk not null unique 'dataset_id'"`
	DatasetName  string           `json:"dataset_name" xorm:"TEXT 'dataset_name'"`
	DateCreated  int64            `json:"date_created" xorm:"INTEGER 'date_created'"`
	DateModified int64            `json:"date_modified" xorm:"INTEGER 'date_modified'"`
	DatasetType  MynahDatasetType `json:"dataset_type" xorm:"TEXT 'dataset_type'"`
}

// MynahDatasetVersionDatasetLink defines a dataset to dataset-version relationship
type MynahDatasetVersionDatasetLink struct {
	ID               int64           `xorm:"pk autoincr"`
	DatasetId        types.MynahUuid `json:"dataset_id" xorm:"varchar(36) not null index 'dataset_id'"`
	DatasetVersionId types.MynahUuid `json:"dataset_version_id" xorm:"varchar(36) not null index 'dataset_version_id'"`
}

// MynahICProcessTaskData defines the result of running a task on an ic dataset
type MynahICProcessTaskData struct {
	//TODO
}

// MynahICDatasetVersion defines a version of an image classification dataset version
type MynahICDatasetVersion struct {
	DatasetVersionId types.MynahUuid           `json:"dataset_version_id" xorm:"varchar(36) pk not null unique 'dataset_version_id'"`
	VersionIndex     int64                     `json:"version_index" xorm:"INTEGER 'version_index'"`
	DateCreated      int64                     `json:"date_created" xorm:"INTEGER 'date_created'"`
	Mean             []float64                 `json:"mean" xorm:"TEXT 'mean'"`
	StdDev           []float64                 `json:"std_dev" xorm:"TEXT 'std_dev'"`
	TaskData         []*MynahICProcessTaskData `json:"task_data,omitempty" xorm:"TEXT 'task_data'"`
}

func init() {
	db.RegisterTables(
		&MynahDataset{},
		&MynahDatasetVersionDatasetLink{},
		&MynahICDatasetVersion{},
	)
}

// CreateMynahDataset creates a new dataset
func CreateMynahDataset(ctx *db.Context, dataset *MynahDataset) error {
	// create the dataset
	affected, err := ctx.GetEngine().Insert(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("dataset (%s) not created (no records affected)", dataset.DatasetId)
	}
	return nil
}
