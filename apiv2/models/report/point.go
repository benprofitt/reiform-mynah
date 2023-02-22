// Copyright (c) 2023 by Reiform. All Rights Reserved.

package report

import (
	"reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/types"
)

// MynahICDatasetReportPoint defines a plotted point and associated info
type MynahICDatasetReportPoint struct {
	FileId        types.MynahUuid        `json:"fileid"`
	X             float64                `json:"x"`
	Y             float64                `json:"y"`
	OriginalClass dataset.MynahClassName `json:"original_class"`
}
