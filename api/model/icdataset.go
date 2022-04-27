// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// ConfidenceVectors for a file
type ConfidenceVectors [][]float64

// MynahICDatasetReportPoint a cartesian point
type MynahICDatasetReportPoint struct {
	X int64 `json:"x"`
	Y int64 `json:"y"`
}

// MynahICDatasetReportBucket classifications for images in class
type MynahICDatasetReportBucket struct {
	Bad        int `json:"bad"`
	Acceptable int `json:"acceptable"`
}

// MynahICDatasetReportImageMetadata defines the metadata associated with an image
type MynahICDatasetReportImageMetadata struct {
	//the version id for the file
	ImageVersionId MynahFileVersionId `json:"image_version_id"`
	//the class
	Class string `json:"class"`
	//whether the image was mislabeled
	Mislabeled bool `json:"mislabeled"`
	//point for display in graph
	Point MynahICDatasetReportPoint `json:"point"`
	//the outlier sets this image is in
	OutlierSets []string `json:"outlier_sets"`
}

// MynahICDatasetReport a mynah image classification report
type MynahICDatasetReport struct {
	//all of the images, in the order to display
	ImageIds []MynahUuid `json:"image_ids"`
	//the images included in this report, map fileid to metadata
	ImageData map[MynahUuid]*MynahICDatasetReportImageMetadata `json:"image_data"`
	//the class breakdown table info, map class to buckets
	Breakdown map[string]*MynahICDatasetReportBucket `json:"breakdown"`
}

// MynahICDatasetFile file data
type MynahICDatasetFile struct {
	//the version id for the file
	ImageVersionId MynahFileVersionId `json:"image_version_id"`
	//the current clas
	CurrentClass string `json:"current_class"`
	//the original class
	OriginalClass string `json:"original_class"`
	//the confidence vectors
	ConfidenceVectors ConfidenceVectors `json:"confidence_vectors"`
	//projections
	Projections map[string][]int `json:"projections"`
}

// MynahICDatasetVersion defines a specific version of the dataset
type MynahICDatasetVersion struct {
	//map of fileid -> file + class info
	Files map[MynahUuid]*MynahICDatasetFile `json:"files"`
	//the unified report
	Report *MynahICDatasetReport `json:"report"`
}

// MynahICDataset Defines a dataset specifically for image classification
type MynahICDataset struct {
	//underlying mynah dataset
	MynahDataset `xorm:"extends"`
	//versions of the dataset
	Versions map[MynahDatasetVersionId]*MynahICDatasetVersion `json:"versions" xorm:"TEXT 'versions'"`
}

// NewICDataset creates a new dataset
func NewICDataset(creator *MynahUser) *MynahICDataset {
	return &MynahICDataset{
		MynahDataset: *NewDataset(creator),
		Versions:     make(map[MynahDatasetVersionId]*MynahICDatasetVersion),
	}
}

// NewICDatasetFile creates a new dataset file record
func NewICDatasetFile() *MynahICDatasetFile {
	return &MynahICDatasetFile{
		ImageVersionId:    LatestVersionId,
		CurrentClass:      "",
		OriginalClass:     "",
		ConfidenceVectors: make(ConfidenceVectors, 0),
		Projections:       make(map[string][]int),
	}
}

// NewMynahICDatasetReport creates a new report
func NewMynahICDatasetReport() *MynahICDatasetReport {
	return &MynahICDatasetReport{
		ImageIds:  make([]MynahUuid, 0),
		ImageData: make(map[MynahUuid]*MynahICDatasetReportImageMetadata),
		Breakdown: make(map[string]*MynahICDatasetReportBucket),
	}
}

// CopyICDatasetFile creates a new dataset file record by copying another
func CopyICDatasetFile(other *MynahICDatasetFile) *MynahICDatasetFile {
	confidenceVectors := make(ConfidenceVectors, 0)
	copy(confidenceVectors, other.ConfidenceVectors)
	projections := make(map[string][]int)
	for key, value := range other.Projections {
		projections[key] = value
	}

	return &MynahICDatasetFile{
		ImageVersionId:    LatestVersionId,
		CurrentClass:      other.CurrentClass,
		OriginalClass:     other.OriginalClass,
		ConfidenceVectors: confidenceVectors,
		Projections:       projections,
	}
}
