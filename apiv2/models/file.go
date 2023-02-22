// Copyright (c) 2023 by Reiform. All Rights Reserved.

package models

import (
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/types"
)

// MynahFile defines an immutable file
type MynahFile struct {
	FileId      types.MynahUuid `json:"file_id" xorm:"varchar(36) pk not null unique 'file_id'"`
	Name        string          `json:"name" xorm:"TEXT 'name'"`
	DateCreated int64           `json:"date_created" xorm:"INTEGER 'date_created'"`
	MimeType    string          `json:"mime_type" xorm:"TEXT 'mime_type'"`
}

// MynahDatasetVersionFileLink defines a dataset-version to file relationship
type MynahDatasetVersionFileLink struct {
	ID               int64           `xorm:"pk autoincr"`
	DatasetVersionId types.MynahUuid `json:"dataset_version_id" xorm:"varchar(36) not null index 'dataset_version_id'"`
	FileId           types.MynahUuid `json:"file_id" xorm:"varchar(36) not null index 'file_id'"`
}

func init() {
	db.RegisterTables(
		&MynahFile{},
		&MynahDatasetVersionFileLink{},
	)
}
