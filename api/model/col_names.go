// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

type MynahColName string

const (
	VersionsColName      MynahColName = "versions"
	ReportsColName       MynahColName = "reports"
	LatestVersionColName MynahColName = "latest_version"
	DateModifiedCol      MynahColName = "date_modified"

	DatasetNameCol MynahColName = "dataset_name"
	NameFirstCol   MynahColName = "name_first"
	NameLastCol    MynahColName = "name_last"
)

type NonUpdatableColName string

const (
	UuidColName  NonUpdatableColName = "uuid"
	OrgIdColName NonUpdatableColName = "org_id"
)

// MynahDatabaseColumns maintains relationship between columns that must be updated together
// Note: a column name **must** be in this map to be updated by the database
var MynahDatabaseColumns = map[MynahColName][]MynahColName{
	VersionsColName: {
		LatestVersionColName,
	},
	ReportsColName: {
		LatestVersionColName,
	},
	LatestVersionColName: {},
	DateModifiedCol:      {},
	DatasetNameCol:       {},
	NameFirstCol:         {},
	NameLastCol:          {},
}

// MynahNonUpdatableDatabaseColumns maintains a set of keys that may not be updated after creation
var MynahNonUpdatableDatabaseColumns = map[NonUpdatableColName]interface{}{
	UuidColName:  nil,
	OrgIdColName: nil,
}
