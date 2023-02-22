// Copyright (c) 2023 by Reiform. All Rights Reserved.

package models

import (
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/types"
)

// MynahDatasetReport defines a dataset version report
type MynahDatasetReport struct {
	ReportId    types.MynahUuid `json:"report_id" xorm:"varchar(36) pk not null unique index 'report_id'"`
	DateCreated int64           `json:"date_created" xorm:"INTEGER 'date_created'"`
	CreatedBy   types.MynahUuid `json:"created_by" xorm:"TEXT 'created_by'"`
}

// MynahICDatasetReportContents defines an image classification report
type MynahICDatasetReportContents struct {
	ID       int64           `json:"-" xorm:"pk autoincr"`
	ReportId types.MynahUuid `json:"report_id" xorm:"varchar(36) not null unique index 'report_id'"`

	// TODO
}

// MynahDatasetVersionReportLink defines a dataset-version to report relationship
type MynahDatasetVersionReportLink struct {
	ID             int64           `xorm:"pk autoincr"`
	DatasetVersion types.MynahUuid `json:"dataset_version_id" xorm:"varchar(36) not null index 'dataset_version_id'"`
	ReportId       types.MynahUuid `json:"report_id" xorm:"varchar(36) not null index 'report_id'"`
}

func init() {
	db.RegisterTables(
		&MynahDatasetReport{},
		&MynahICDatasetReportContents{},
		&MynahDatasetVersionReportLink{},
	)
}
