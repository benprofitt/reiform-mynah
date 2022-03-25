// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahICDiagnosisReportPoint a cartesian point
type MynahICDiagnosisReportPoint struct {
	X int64 `json:"x"`
	Y int64 `json:"y"`
}

// MynahICDiagnosisReportBucket classifications for images in class
type MynahICDiagnosisReportBucket struct {
	Bad        int `json:"bad"`
	Acceptable int `json:"acceptable"`
}

// MynahICDiagnosisReportImageMetadata defines the metadata associated with an image
type MynahICDiagnosisReportImageMetadata struct {
	//the class
	Class string `json:"class"`
	//whether the image was mislabeled
	Mislabeled bool `json:"mislabeled"`
	//point for display in graph
	Point MynahICDiagnosisReportPoint `json:"point"`
	//the outlier sets this image is in
	OutlierSets []string `json:"outlier_sets"`
}

// MynahICDiagnosisReport a mynah image classification diagnosis report
type MynahICDiagnosisReport struct {
	//underlying mynah dataset
	MynahReport `json:"report" xorm:"extends"`
	//all of the images, in the order to display
	ImageIds []string `json:"image_ids" xorm:"TEXT 'image_ids'"`
	//the images included in this report, map fileid to metadata
	ImageData map[string]*MynahICDiagnosisReportImageMetadata `json:"image_data" xorm:"TEXT 'image_data'"`
	//the class breakdown table info, map class to buckets
	Breakdown map[string]*MynahICDiagnosisReportBucket `json:"breakdown" xorm:"TEXT 'breakdown'"`
}
