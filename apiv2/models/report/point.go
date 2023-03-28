// Copyright (c) 2023 by Reiform. All Rights Reserved.

package report

import (
	"reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/models/types"
)

// MynahICDatasetVersionReportPoint defines a plotted point and associated info
type MynahICDatasetVersionReportPoint struct {
	ID       int64                  `json:"-" xorm:"pk autoincr"`
	ReportId types.MynahUuid        `json:"report_id" xorm:"varchar(36) not null index 'report_id'"`
	FileId   types.MynahUuid        `json:"file_id" xorm:"varchar(36) not null index 'file_id'"`
	Class    dataset.MynahClassName `json:"class" xorm:"TEXT not null index 'class'"`
	Point    []float64              `json:"point" xorm:"TEXT 'point'"`
}

func init() {
	db.RegisterTables(
		&MynahICDatasetVersionReportPoint{},
	)
}
