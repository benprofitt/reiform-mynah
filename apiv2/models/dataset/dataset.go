// Copyright (c) 2023 by Reiform. All Rights Reserved.

package dataset

import (
	"fmt"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/types"
)

type MynahDatasetType string
type MynahClassName string
type ConfidenceVectors [][]float64
type MynahICProcessTaskType string

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
	CreatedBy    types.MynahUuid  `json:"created_by" xorm:"TEXT 'created_by'"`
}

// MynahICDatasetVersion defines a version of an image classification dataset
type MynahICDatasetVersion struct {
	DatasetVersionId types.MynahUuid           `json:"dataset_version_id" xorm:"varchar(36) pk not null unique 'dataset_version_id'"`
	DatasetId        types.MynahUuid           `json:"dataset_id" xorm:"varchar(36) not null index 'dataset_id'"`
	AncestorId       types.MynahUuid           `json:"ancestor_id" xorm:"varchar(36) 'ancestor_id'"`
	DateCreated      int64                     `json:"date_created" xorm:"INTEGER 'date_created'"`
	Mean             []float64                 `json:"mean" xorm:"TEXT 'mean'"`
	StdDev           []float64                 `json:"std_dev" xorm:"TEXT 'std_dev'"`
	TaskData         []*MynahICProcessTaskData `json:"task_data" xorm:"TEXT 'task_data'"`
	CreatedBy        types.MynahUuid           `json:"created_by" xorm:"TEXT 'created_by'"`
}

func init() {
	db.RegisterTables(
		&MynahDataset{},
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

// GetMynahDataset gets a dataset by id
func GetMynahDataset(ctx *db.Context, datasetId types.MynahUuid) (*MynahDataset, bool, error) {
	dataset := MynahDataset{
		DatasetId: datasetId,
	}
	found, err := ctx.GetEngine().Get(&dataset)
	if err != nil {
		return nil, false, err
	}
	return &dataset, found, nil
}

// ListMynahDatasets returns a list of datasets
func ListMynahDatasets(ctx *db.Context, opts *db.PaginationOptions) (*db.Paginated[*MynahDataset], error) {
	datasets := make([]*MynahDataset, 0, opts.PageSize)

	count, err := ctx.GetEngine().OrderBy("dataset_id").Limit(opts.PageSize, opts.Page*opts.PageSize).FindAndCount(&datasets)
	if err != nil {
		return nil, err
	}

	return &db.Paginated[*MynahDataset]{
		Page:     opts.Page,
		PageSize: opts.PageSize,
		Total:    count,
		Contents: datasets,
	}, nil
}
