// Copyright (c) 2023 by Reiform. All Rights Reserved.

package file

import (
	"fmt"
	"reiform.com/mynah-api/models"
	"reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/models/types"
)

// MynahFile defines an immutable file
type MynahFile struct {
	FileId      types.MynahUuid `json:"file_id" xorm:"varchar(36) pk not null unique 'file_id'"`
	Name        string          `json:"name" xorm:"TEXT 'name'"`
	DateCreated int64           `json:"date_created" xorm:"INTEGER 'date_created'"`
	CreatedBy   types.MynahUuid `json:"created_by" xorm:"TEXT 'created_by'"`
}

// MynahICDatasetVersionFile defines information relative to a file for a given image classification dataset version
type MynahICDatasetVersionFile struct {
	ID                int64                     `json:"-" xorm:"pk autoincr"`
	DatasetVersionId  types.MynahUuid           `json:"dataset_version_id" xorm:"varchar(36) not null index 'dataset_version_id'"`
	FileId            types.MynahUuid           `json:"file_id" xorm:"varchar(36) not null index 'file_id'"`
	Class             dataset.MynahClassName    `json:"class" xorm:"TEXT 'class'"`
	ConfidenceVectors dataset.ConfidenceVectors `json:"confidence_vectors" xorm:"TEXT 'confidence_vectors'"`
	Projections       struct {
		ProjectionLabelFullEmbeddingConcatenation []float64 `json:"projection_label_full_embedding_concatenation"`
		ProjectionLabelReducedEmbedding           []float64 `json:"projection_label_reduced_embedding"`
		ProjectionLabelReducedEmbeddingPerClass   []float64 `json:"projection_label_reduced_embedding_per_class"`
		ProjectionLabel2dPerClass                 []float64 `json:"projection_label_2d_per_class"`
		ProjectionLabel2d                         []float64 `json:"projection_label_2d"`
	} `json:"projections" xorm:"TEXT 'projections'"`
	Mean   []float64 `json:"mean" xorm:"TEXT 'mean'"`
	StdDev []float64 `json:"std_dev" xorm:"TEXT 'std_dev'"`
}

const updateBatchSize = 500

func init() {
	db.RegisterTables(
		&MynahFile{},
		&MynahICDatasetVersionFile{},
	)
}

// CreateMynahFile creates a new file
func CreateMynahFile(ctx *db.Context, file *MynahFile) error {
	affected, err := ctx.Engine().Insert(file)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("file (%s) not created (no records affected)", file.FileId)
	}
	return nil
}

// AssignMynahICDatasetClasses assigns classes for files in a given dataset version
func AssignMynahICDatasetClasses(ctx *db.Context, datasetVersionId types.MynahUuid, assignments map[types.MynahUuid]dataset.MynahClassName) error {
	return ctx.NewTransaction(func(tx *db.Context) error {
		kvs := models.KeysVals(assignments, func(fileId types.MynahUuid, className dataset.MynahClassName) *MynahICDatasetVersionFile {
			return &MynahICDatasetVersionFile{
				DatasetVersionId: datasetVersionId,
				Class:            className,
			}
		})

		// update in batches
		for i := 0; i < len(assignments); i += updateBatchSize {
			batch := kvs[i:models.Min(i+updateBatchSize, len(assignments))]

			if _, err := tx.Engine().Cols("class").Update(&batch); err != nil {
				return fmt.Errorf("dataset (version=%s) file classes not updated (batch_offset=%d,batch_size=%d,total=%d): %s",
					datasetVersionId,
					i,
					updateBatchSize,
					len(assignments),
					err)
			}
		}
		return nil
	})
}
