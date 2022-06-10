// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// ConfidenceVectors for a file
type ConfidenceVectors [][]float64

// MynahICProcessTaskType is a task type name
type MynahICProcessTaskType string

const (
	ICProcessDiagnoseMislabeledImagesTask   MynahICProcessTaskType = "ic::diagnose::mislabeled_images"
	ICProcessCorrectMislabeledImagesTask    MynahICProcessTaskType = "ic::correct::mislabeled_images"
	ICProcessDiagnoseClassSplittingTask     MynahICProcessTaskType = "ic::diagnose::class_splitting"
	ICProcessCorrectClassSplittingTask      MynahICProcessTaskType = "ic::correct::class_splitting"
	ICProcessDiagnoseLightingConditionsTask MynahICProcessTaskType = "ic::diagnose::lighting_conditions"
	ICProcessCorrectLightingConditionsTask  MynahICProcessTaskType = "ic::correct::lighting_conditions"
	ICProcessDiagnoseImageBlurTask          MynahICProcessTaskType = "ic::diagnose::image_blur"
	ICProcessCorrectImageBlurTask           MynahICProcessTaskType = "ic::correct::image_blur"
)

// MynahICDatasetFormatType is the dataset format type identifier
type MynahICDatasetFormatType string

const (
	ICDatasetFolderFormat MynahICDatasetFormatType = "ic::folder_format"
)

// MynahICDatasetFormatMetadata defines a dataset format
type MynahICDatasetFormatMetadata interface {
	// DatasetFileIterator takes a map from fileid to original filename and takes a function that is called for each file to export
	// the handler takes a file, the version to export, and the path to write the file to in the zip
	DatasetFileIterator(*MynahICDatasetVersion, map[MynahUuid]string, func(fileId MynahUuid, fileVersion MynahFileVersionId, filePath string) error) error
	// GenerateArtifacts takes a handler that is called for each additional artifact the dataset export
	// generates. THis includes the file contents and the path to write to in the zip archive
	// Note: filePath includes the filename
	GenerateArtifacts(*MynahICDatasetVersion, func(fileContents []byte, filePath string) error) error
}

// MynahICDatasetFormat defines the import/export format for a dataset
type MynahICDatasetFormat struct {
	// the format type
	Type MynahICDatasetFormatType `json:"type"`
	// the metadata
	Metadata MynahICDatasetFormatMetadata `json:"metadata"`
}

// MynahICProcessTaskMetadata defines metadata specific to some task
type MynahICProcessTaskMetadata interface {
	// ApplyToDataset makes changes to the dataset based on this metadata
	ApplyToDataset(*MynahICDatasetVersion, MynahICProcessTaskType) error
	// ApplyToReport makes changes to the report based on this metadata
	ApplyToReport(*MynahICDatasetReport, MynahICProcessTaskType) error
}

// MynahICProcessTaskDiagnoseMislabeledImagesMetadata is metadata for
//diagnosing mislabeled images task response
type MynahICProcessTaskDiagnoseMislabeledImagesMetadata struct {
	//ids of outlier images
	Outliers []MynahUuid `json:"outliers"`
}

// MynahICProcessTaskCorrectMislabeledImagesMetadata is metadata for
//correcting mislabeled images task response
type MynahICProcessTaskCorrectMislabeledImagesMetadata struct {
	//removed images by id
	Removed []MynahUuid `json:"removed"`
	//corrected images by id
	Corrected []MynahUuid `json:"corrected"`
}

// MynahICProcessTaskDiagnoseClassSplittingMetadata is metadata for
//diagnosing class splitting task response
type MynahICProcessTaskDiagnoseClassSplittingMetadata struct {
	//map class to list of fileid lists
	PredictedClassSplits map[MynahClassName][][]MynahUuid `json:"predicted_class_splits"`
}

// MynahICProcessTaskCorrectClassSplittingMetadata is metadata for
//correcting class splitting task response
type MynahICProcessTaskCorrectClassSplittingMetadata struct {
	//map from class name to map from class split name to list of fileids
	ActualClassSplits map[MynahClassName]map[string][]MynahUuid `json:"actual_class_splits"`
}

// MynahICProcessTaskDiagnoseLightingConditionsMetadata is metadata for
//diagnosing lighting conditions task response
type MynahICProcessTaskDiagnoseLightingConditionsMetadata struct {
	//bright images by id
	Bright []MynahUuid `json:"bright"`
	//dark images by id
	Dark []MynahUuid `json:"dark"`
}

// MynahICProcessTaskCorrectLightingConditionsMetadata is metadata for
//correcting lighting conditions task response
type MynahICProcessTaskCorrectLightingConditionsMetadata struct {
	//the uuids of removed images
	Removed []MynahUuid `json:"removed"`
	//the uuids of corrected images
	Corrected []MynahUuid `json:"corrected"`
}

// MynahICProcessTaskDiagnoseImageBlurMetadata is metadata for
//diagnosing image blur task response
type MynahICProcessTaskDiagnoseImageBlurMetadata struct {
}

// MynahICProcessTaskCorrectImageBlurMetadata is metadata for
//correcting image blur task response
type MynahICProcessTaskCorrectImageBlurMetadata struct {
}

// MynahICProcessTaskData defines the result of running a task on an ic dataset
type MynahICProcessTaskData struct {
	//the type of the task
	Type MynahICProcessTaskType `json:"type"`
	//the metadata associated with the task
	Metadata MynahICProcessTaskMetadata `json:"metadata"`
}

// MynahICDatasetFile defines per file metadata for the dataset
type MynahICDatasetFile struct {
	//the version id for the file
	ImageVersionId MynahFileVersionId `json:"image_version_id"`
	//the current clas
	CurrentClass MynahClassName `json:"current_class"`
	//the original class
	OriginalClass MynahClassName `json:"original_class"`
	//the confidence vectors
	ConfidenceVectors ConfidenceVectors `json:"confidence_vectors"`
	//projections
	Projections map[MynahClassName][]int `json:"projections"`
	//the mean of the this file's channels
	Mean []float64 `json:"mean"`
	//the stddev of this file's channels
	StdDev []float64 `json:"std_dev"`
}

// MynahICDatasetVersion defines a specific version of the dataset
type MynahICDatasetVersion struct {
	//map of fileid -> file + class info
	Files map[MynahUuid]*MynahICDatasetFile `json:"files"`
	//the mean of the channels of images in the dataset
	Mean []float64 `json:"mean"`
	//the stddev of the channels of images in the dataset
	StdDev []float64 `json:"std_dev"`
	// task metadata generated by ic process (may be used by subsequent ic process runs)
	TaskData []*MynahICProcessTaskData `json:"task_data,omitempty"`
}

// MynahICDataset Defines a dataset specifically for image classification
type MynahICDataset struct {
	//underlying mynah dataset
	MynahDataset `xorm:"extends"`
	//versions of the dataset
	Versions map[MynahDatasetVersionId]*MynahICDatasetVersion `json:"versions" xorm:"TEXT 'versions'"`
	//reports for this dataset by version (references a binobject)
	Reports map[MynahDatasetVersionId]MynahUuid `json:"-" xorm:"TEXT 'reports'"`
	//the latest version
	LatestVersion MynahDatasetVersionId `json:"latest_version" xorm:"TEXT 'latest_version'"`
	//the format
	Format MynahICDatasetFormat `json:"-" xorm:"TEXT 'format'"`
}

// NewICDataset creates a new dataset
func NewICDataset(creator *MynahUser) *MynahICDataset {
	return &MynahICDataset{
		MynahDataset:  *NewDataset(creator),
		Versions:      make(map[MynahDatasetVersionId]*MynahICDatasetVersion),
		Reports:       make(map[MynahDatasetVersionId]MynahUuid),
		LatestVersion: "0",
		Format: MynahICDatasetFormat{
			Type:     ICDatasetFolderFormat,
			Metadata: &MynahICDatasetFolderFormat{},
		},
	}
}

// NewICDatasetVersion creates a new dataset version
func NewICDatasetVersion() *MynahICDatasetVersion {
	return &MynahICDatasetVersion{
		Files:    make(map[MynahUuid]*MynahICDatasetFile),
		Mean:     make([]float64, 0),
		StdDev:   make([]float64, 0),
		TaskData: make([]*MynahICProcessTaskData, 0),
	}
}

// NewICDatasetFile creates a new dataset file record
func NewICDatasetFile() *MynahICDatasetFile {
	return &MynahICDatasetFile{
		ImageVersionId:    LatestVersionId,
		CurrentClass:      "",
		OriginalClass:     "",
		ConfidenceVectors: make(ConfidenceVectors, 0),
		Projections:       make(map[MynahClassName][]int),
		Mean:              make([]float64, 0),
		StdDev:            make([]float64, 0),
	}
}

// CopyICDatasetFile creates a new dataset file record by copying another
// Note: updates to 'latest' tag
func CopyICDatasetFile(other *MynahICDatasetFile) *MynahICDatasetFile {
	var confidenceVectors ConfidenceVectors
	copy(confidenceVectors, other.ConfidenceVectors)

	projections := make(map[MynahClassName][]int)
	for key, value := range other.Projections {
		projections[key] = value
	}

	var mean []float64
	copy(mean, other.Mean)

	var stdDev []float64
	copy(stdDev, other.StdDev)

	return &MynahICDatasetFile{
		ImageVersionId:    LatestVersionId,
		CurrentClass:      other.CurrentClass,
		OriginalClass:     other.OriginalClass,
		ConfidenceVectors: confidenceVectors,
		Projections:       projections,
		Mean:              mean,
		StdDev:            stdDev,
	}
}
