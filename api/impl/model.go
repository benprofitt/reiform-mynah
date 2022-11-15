// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

import (
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
)

// ImplProvider provider interface
type ImplProvider interface {
	// GetMynahImplVersion get the current version
	GetMynahImplVersion() (*VersionResponse, error)
	// ICProcessJob start ic diagnosis on some dataset
	ICProcessJob(*model.MynahUser, model.MynahUuid, []model.MynahICProcessTaskType) error
	// BatchImageMetadata gets image metadata. Note: this will overwrite any metadata in the specified file version
	BatchImageMetadata(*model.MynahUser, storage.MynahLocalFileSet) error
}

// VersionResponse mynah python version response
type VersionResponse struct {
	//the version returned by python
	Version string `json:"version"`
}

// ICProcessJobRequestFile format of files for a start diagnosis request
type ICProcessJobRequestFile struct {
	//the uuid for the file
	Uuid model.MynahUuid `json:"uuid"`
	//the width of the image
	Width int64 `json:"width"`
	//the height of the image
	Height int64 `json:"height"`
	//the number of channels in the image
	Channels int64 `json:"channels"`
	//the mean of the channels in this image
	Mean []float64 `json:"mean"`
	//the stddev of the channels in this image
	StdDev []float64 `json:"std_dev"`
}

// ICProcessJobRequestTask defines a task to perform
type ICProcessJobRequestTask struct {
	//the type of the task
	Type model.MynahICProcessTaskType `json:"type"`
}

// ICProcessJobRequest request format for a diagnosis job
type ICProcessJobRequest struct {
	ConfigParams struct {
		ModelsPath string `json:"models_path"`
	} `json:"config_params"`
	//the dataset to operate on
	Dataset struct {
		//the uuid of the dataset
		DatasetUuid model.MynahUuid `json:"uuid"`
		//all classes in the dataset
		Classes []model.MynahClassName `json:"classes"`
		//the mean of the channels of images in the dataset
		Mean []float64 `json:"mean"`
		//the stddev of the channels of images in the dataset
		StdDev []float64 `json:"std_dev"`
		//map by class to map by temp path
		ClassFiles map[model.MynahClassName]map[string]*ICProcessJobRequestFile `json:"class_files"`
	} `json:"dataset"`
	//tasks to perform
	Tasks []ICProcessJobRequestTask `json:"tasks"`
	//tasks performed previously
	PreviousResults []*model.MynahICProcessTaskData `json:"previous_results"`
}

// ICProcessJobResponseFile file data returned in response
type ICProcessJobResponseFile struct {
	//the uuid of the file
	Uuid model.MynahUuid `json:"uuid"`
	//the current class for this file
	CurrentClass model.MynahClassName `json:"current_class"`
	//projections
	Projections map[model.MynahClassName][]int `json:"projections"`
	//confidence vectors
	ConfidenceVectors model.ConfidenceVectors `json:"confidence_vectors"`
	//the mean of the channels of this image
	Mean []float64 `json:"mean"`
	//the stddev of the channels this image
	StdDev []float64 `json:"std_dev"`
}

// ICProcessJobResponse response format for a diagnosis job
type ICProcessJobResponse struct {
	//the dataset response
	Dataset struct {
		//the uuid of the dataset
		DatasetUuid model.MynahUuid `json:"uuid"`
		//all classes in the dataset
		Classes []model.MynahClassName `json:"classes"`
		//the mean of the channels of images in the dataset
		Mean []float64 `json:"mean"`
		//the stddev of the channels of images in the dataset
		StdDev []float64 `json:"std_dev"`
		//map by class to map by temp path
		ClassFiles map[model.MynahClassName]map[string]*ICProcessJobResponseFile `json:"class_files"`
	} `json:"dataset"`
	//the tasks in the dataset
	Tasks []model.MynahICProcessTaskData `json:"tasks"`
}

// ImageMetadataRequestLocalFile defines a file that is part of the batch
type ImageMetadataRequestLocalFile struct {
	// the uuid of the file
	Uuid model.MynahUuid `json:"uuid"`
	// the path to the file locally
	Path string `json:"path"`
}

// ImageMetadataRequest request type for image metadata
type ImageMetadataRequest struct {
	// the images to process
<<<<<<< HEAD
	Images []ImageMetadataRequestLocalFile `json:"images"`
=======
	Images []*ImageMetadataRequestLocalFile `json:"images"`
>>>>>>> develop
}

// ImageMetadataResponseFileData defines the metadata for a single file
type ImageMetadataResponseFileData struct {
	//the number of channels in an image
	Channels int `json:"channels"`
	//the height of the image
	Height int `json:"height"`
	//the width of the image
	Width int `json:"width"`
	//the mean of the channels
	Mean []float64 `json:"mean"`
	//the stdev of the channels
	StdDev []float64 `json:"std_dev"`
}

// ImageMetadataResponse response type for image metadata
type ImageMetadataResponse struct {
	// the images to batch process
<<<<<<< HEAD
	Images map[model.MynahUuid]ImageMetadataResponseFileData `json:"images"`
=======
	Images map[model.MynahUuid]*ImageMetadataResponseFileData `json:"images"`
>>>>>>> develop
}
