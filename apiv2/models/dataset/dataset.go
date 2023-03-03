// Copyright (c) 2023 by Reiform. All Rights Reserved.

package dataset

import (
	"fmt"
	"github.com/gin-gonic/gin/binding"
	"github.com/go-playground/validator/v10"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/models/types"
	"reiform.com/mynah-api/services/log"
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

// MynahDatasetVersionRef defines a mynah dataset version ordering
type MynahDatasetVersionRef struct {
	DatasetVersionId types.MynahUuid `json:"dataset_version_id" xorm:"varchar(36) 'dataset_version_id'"`
	AncestorId       types.MynahUuid `json:"ancestor_id" xorm:"varchar(36) 'ancestor_id'"`
}

func init() {
	db.RegisterTables(
		&MynahDataset{},
		&MynahICDatasetVersion{},
	)

	if v, ok := binding.Validator.Engine().(*validator.Validate); ok {
		if err := v.RegisterValidation("mynah_dataset_type", ValidateMynahDatasetType); err != nil {
			log.Error("failed to register 'mynah_dataset_type' validation")
		}
	}
}

func (s MynahDatasetType) IsValid() bool {
	switch s {
	case ImageClassificationDataset:
		return true
	}
	return false
}

// ValidateMynahDatasetType checks the validity of a dataset type
func ValidateMynahDatasetType(fl validator.FieldLevel) bool {
	value := fl.Field().Interface().(MynahDatasetType)
	return value.IsValid()
}

// CreateMynahDataset creates a new dataset
func CreateMynahDataset(ctx *db.Context, dataset *MynahDataset) error {
	// create the dataset
	affected, err := ctx.Engine().Insert(dataset)
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
	found, err := ctx.Engine().Get(&dataset)
	if err != nil {
		return nil, false, err
	}
	return &dataset, found, nil
}

// ListMynahDatasets returns a list of datasets
func ListMynahDatasets(ctx *db.Context, opts *db.PaginationOptions) (*db.Paginated[*MynahDataset], error) {
	datasets := make([]*MynahDataset, 0, opts.PageSize)

	count, err := ctx.Engine().OrderBy("dataset_id").Limit(opts.PageSize, opts.Page*opts.PageSize).FindAndCount(&datasets)
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

// CreateMynahICDatasetVersion creates a new version of an ic dataset
func CreateMynahICDatasetVersion(ctx *db.Context, datasetVersion *MynahICDatasetVersion) error {
	// create the dataset version
	affected, err := ctx.Engine().Insert(datasetVersion)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("dataset version (dataset_id=%s, version=%s) not created (no records affected)",
			datasetVersion.DatasetId,
			datasetVersion.DatasetVersionId)
	}
	return nil
}

// GetMynahICDatasetVersion gets an image classification dataset version
func GetMynahICDatasetVersion(ctx *db.Context, datasetId, versionId types.MynahUuid) (*MynahICDatasetVersion, bool, error) {
	datasetVersion := MynahICDatasetVersion{
		DatasetVersionId: versionId,
		DatasetId:        datasetId,
	}
	found, err := ctx.Engine().Get(&datasetVersion)
	if err != nil {
		return nil, false, err
	}
	return &datasetVersion, found, nil
}

// ListMynahICDatasetVersionRefs gets the version refs for a dataset
func ListMynahICDatasetVersionRefs(ctx *db.Context, datasetId types.MynahUuid) ([]*MynahDatasetVersionRef, error) {
	versions := make([]*MynahDatasetVersionRef, 0)
	return versions, ctx.Engine().
		Table("mynah_i_c_dataset_version").
		Cols("dataset_version_id", "ancestor_id").
		Where("`dataset_id` = ?", datasetId).
		Find(&versions)
}

// ListMynahICDatasets lists versions for a dataset
func ListMynahICDatasets(ctx *db.Context, datasetId types.MynahUuid, opts *db.PaginationOptions) (*db.Paginated[*MynahICDatasetVersion], error) {
	datasetVersions := make([]*MynahICDatasetVersion, 0, opts.PageSize)

	count, err := ctx.Engine().
		Where("`dataset_id` = ?", datasetId).
		OrderBy("dataset_version_id").
		Limit(opts.PageSize, opts.Page*opts.PageSize).
		FindAndCount(&datasetVersions)
	if err != nil {
		return nil, err
	}

	return &db.Paginated[*MynahICDatasetVersion]{
		Page:     opts.Page,
		PageSize: opts.PageSize,
		Total:    count,
		Contents: datasetVersions,
	}, nil
}
