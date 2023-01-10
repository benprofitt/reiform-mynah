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
