// Copyright (c) 2023 by Reiform. All Rights Reserved.

package models

import (
	"reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/types"
)

// MynahFile defines an immutable file
type MynahFile struct {
	FileId      types.MynahUuid `json:"file_id" xorm:"varchar(36) pk not null unique 'file_id'"`
	Name        string          `json:"name" xorm:"TEXT 'name'"`
	DateCreated int64           `json:"date_created" xorm:"INTEGER 'date_created'"`
	MimeType    string          `json:"mime_type" xorm:"TEXT 'mime_type'"`
	CreatedBy   types.MynahUuid `json:"created_by" xorm:"TEXT 'created_by'"`
}

// MynahICDatasetVersionFile defines information relative to a file for a given image classification dataset version
type MynahICDatasetVersionFile struct {
	ID                int64                     `xorm:"pk autoincr"`
	DatasetVersionId  types.MynahUuid           `json:"dataset_version_id" xorm:"varchar(36) not null index 'dataset_version_id'"`
	FileId            types.MynahUuid           `json:"file_id" xorm:"varchar(36) not null index 'file_id'"`
	CurrentClass      dataset.MynahClassName    `json:"current_class" xorm:"TEXT 'current_class'"`
	OriginalClass     dataset.MynahClassName    `json:"original_class" xorm:"TEXT 'original_class'"`
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

func init() {
	db.RegisterTables(
		&MynahFile{},
		&MynahICDatasetVersionFile{},
	)
}
