// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// ConfidenceVectors for a file
type ConfidenceVectors [][]float64

// MynahICProcessTaskType is a task type name
type MynahICProcessTaskType string

const (
	ICProcessDiagnoseMislabeledImagesTask MynahICProcessTaskType = "ic::diagnose::mislabeled_images"
	ICProcessCorrectMislabeledImagesTask  MynahICProcessTaskType = "ic::correct::mislabeled_images"
	ICProcessDiagnoseClassSplittingTask   MynahICProcessTaskType = "ic::diagnose::class_splitting"
	ICProcessCorrectClassSplittingTask    MynahICProcessTaskType = "ic::correct::class_splitting"
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
	// ApplyChanges applies these changes to the dataset and the report
	ApplyChanges(*MynahICDatasetVersion, *MynahICDatasetReport, MynahICProcessTaskType) error
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
	ActualClassSplits map[MynahClassName]map[MynahClassName][]MynahUuid `json:"actual_class_splits"`
}

// MynahICProcessTaskData defines the result of running a task on an ic dataset
type MynahICProcessTaskData struct {
	//the type of the task
	Type MynahICProcessTaskType `json:"type"`
	//the metadata associated with the task
	Metadata MynahICProcessTaskMetadata `json:"metadata"`
}

// MynahICDatasetFileProjections defines projections for a dataset file
type MynahICDatasetFileProjections struct {
	ProjectionLabelFullEmbeddingConcatenation []float64 `json:"projection_label_full_embedding_concatenation"`
	ProjectionLabelReducedEmbedding           []float64 `json:"projection_label_reduced_embedding"`
	ProjectionLabelReducedEmbeddingPerClass   []float64 `json:"projection_label_reduced_embedding_per_class"`
	ProjectionLabel2dPerClass                 []float64 `json:"projection_label_2d_per_class"`
	ProjectionLabel2d                         []float64 `json:"projection_label_2d"`
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
	Projections *MynahICDatasetFileProjections `json:"projections"`
	//the mean of this file's channels
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

// MynahICDatasetReportMetadata contains metadata about a report
type MynahICDatasetReportMetadata struct {
	// id for requesting report contents
	DataId MynahUuid `json:"data_id"`
	// the date the report was created
	DateCreated int64 `json:"date_created"`
	// the tasks that were run
	Tasks []MynahICProcessTaskType `json:"tasks"`
}

// MynahICDataset Defines a dataset specifically for image classification
type MynahICDataset struct {
	//underlying mynah dataset
	MynahDataset `xorm:"extends"`
	//versions of the dataset
	Versions map[MynahDatasetVersionId]*MynahICDatasetVersion `json:"versions" xorm:"TEXT 'versions'"`
	//reports for this dataset by version (references a binobject)
	Reports map[MynahDatasetVersionId]*MynahICDatasetReportMetadata `json:"reports" xorm:"TEXT 'reports'"`
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
		Reports:       make(map[MynahDatasetVersionId]*MynahICDatasetReportMetadata),
		LatestVersion: "0",
		Format: MynahICDatasetFormat{
			Type:     ICDatasetFolderFormat,
			Metadata: &MynahICDatasetFolderFormat{},
		},
	}
}

// NewMynahICDatasetFileProjections creates a new set of projections
func NewMynahICDatasetFileProjections() *MynahICDatasetFileProjections {
	return &MynahICDatasetFileProjections{
		ProjectionLabelFullEmbeddingConcatenation: make([]float64, 0),
		ProjectionLabelReducedEmbedding:           make([]float64, 0),
		ProjectionLabelReducedEmbeddingPerClass:   make([]float64, 0),
		ProjectionLabel2dPerClass:                 make([]float64, 0),
		ProjectionLabel2d:                         make([]float64, 0),
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
		Projections:       NewMynahICDatasetFileProjections(),
		Mean:              make([]float64, 0),
		StdDev:            make([]float64, 0),
	}
}

// CopyICDatasetFile creates a new dataset file record by copying another
// Note: updates to 'latest' tag
func CopyICDatasetFile(other *MynahICDatasetFile) *MynahICDatasetFile {
	confidenceVectors := make(ConfidenceVectors, 0)
	for _, v := range other.ConfidenceVectors {
		confidenceVectors = append(confidenceVectors, append([]float64(nil), v...))
	}

	return &MynahICDatasetFile{
		ImageVersionId:    LatestVersionId,
		CurrentClass:      other.CurrentClass,
		OriginalClass:     other.OriginalClass,
		ConfidenceVectors: confidenceVectors,
		Projections: &MynahICDatasetFileProjections{
			ProjectionLabelFullEmbeddingConcatenation: append([]float64(nil), other.Projections.ProjectionLabelFullEmbeddingConcatenation...),
			ProjectionLabelReducedEmbedding:           append([]float64(nil), other.Projections.ProjectionLabelReducedEmbedding...),
			ProjectionLabelReducedEmbeddingPerClass:   append([]float64(nil), other.Projections.ProjectionLabelReducedEmbeddingPerClass...),
			ProjectionLabel2dPerClass:                 append([]float64(nil), other.Projections.ProjectionLabel2dPerClass...),
			ProjectionLabel2d:                         append([]float64(nil), other.Projections.ProjectionLabel2d...),
		},
		Mean:   append([]float64(nil), other.Mean...),
		StdDev: append([]float64(nil), other.StdDev...),
	}
}
