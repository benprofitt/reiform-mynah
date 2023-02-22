// Copyright (c) 2023 by Reiform. All Rights Reserved.

package report

import (
	"reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/types"
)

// MynahICDatasetVersionReport defines an ic dataset version report
type MynahICDatasetVersionReport struct {
	ReportId         types.MynahUuid                                         `json:"report_id" xorm:"varchar(36) pk not null unique 'report_id'"`
	DatasetVersionId types.MynahUuid                                         `json:"dataset_version_id" xorm:"varchar(36) not null index 'dataset_version_id'"`
	DateCreated      int64                                                   `json:"date_created" xorm:"INTEGER 'date_created'"`
	CreatedBy        types.MynahUuid                                         `json:"created_by" xorm:"TEXT 'created_by'"`
	Tasks            []*MynahICProcessTaskReportData                         `json:"tasks" xorm:"TEXT 'tasks'"`
	Points           map[dataset.MynahClassName][]*MynahICDatasetReportPoint `json:"points"`
}

func init() {
	db.RegisterTables(
		&MynahICDatasetVersionReport{},
	)
}
