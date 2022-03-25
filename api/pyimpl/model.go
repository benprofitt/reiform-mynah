// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"reiform.com/mynah/model"
)

const MislabeledTaskName = "mislabeled_images"

// PyImplProvider provider interface
type PyImplProvider interface {
	// GetMynahImplVersion get the current version
	GetMynahImplVersion() (*VersionResponse, error)
	// ICDiagnosisJob start ic diagnosis
	ICDiagnosisJob(*model.MynahUser, *ICDiagnosisJobRequest) (*ICDiagnosisJobResponse, error)
	// ImageMetadata get image metadata
	ImageMetadata(*model.MynahUser, *ImageMetadataRequest) (*ImageMetadataResponse, error)
}

// VersionResponse mynah python version response
type VersionResponse struct {
	//the version returned by python
	Version string `json:"version"`
}

// ICDiagnosisJobRequestFile format of files for a start diagnosis request
type ICDiagnosisJobRequestFile struct {
	//the uuid for the file
	Uuid string `json:"uuid"`
	//the width of the image
	Width int64 `json:"width"`
	//the height of the image
	Height int64 `json:"height"`
	//the number of channels in the image
	Channels int64 `json:"channels"`
}

// ICDiagnosisJobRequest request format for a diagnosis job
type ICDiagnosisJobRequest struct {
	Auto bool `json:"auto"`
	//the uuid of the project
	ProjectUuid string `json:"project_uuid"`
	//the dataset to operate on
	Dataset struct {
		//all classes in the dataset
		Classes []string `json:"classes"`
		//map by class to map by temp path
		ClassFiles map[string]map[string]ICDiagnosisJobRequestFile `json:"class_files"`
	} `json:"dataset"`
}

// ICDiagnosisJobResponseFile file data returned in response
type ICDiagnosisJobResponseFile struct {
	//the uuid of the file
	Uuid string `json:"uuid"`
	//the current class for this file
	CurrentClass string `json:"current_class"`
	//the original class for this file
	OriginalClass string `json:"original_class"`
	//the width of this file
	Width int `json:"width"`
	//the height of this file
	Height int `json:"height"`
	//the channels in this file
	Channels int `json:"channels"`
	//projections
	Projections map[string][]int `json:"projections"`
	//confidence vectors
	ConfidenceVectors model.ConfidenceVectors `json:"confidence_vectors"`
}

// ICDiagnosisJobResponse response format for a diagnosis job
type ICDiagnosisJobResponse struct {
	//the uuid of the project
	ProjectUuid string `json:"project_uuid"`
	//the tasks in the project
	Tasks []struct {
		//the task name
		Name string `json:"name"`
		//the datasets in the task
		Datasets struct {
			//the outliers in the dataset
			Outliers struct {
				//classes
				Classes []string `json:"classes"`
				//map from class to map by fileid to file data
				ClassFiles map[string]map[string]ICDiagnosisJobResponseFile `json:"class_files"`
			} `json:"outliers"`
			//the inliers in the dataset
			Inliers struct {
				//classes
				Classes []string `json:"classes"`
				//map from class to map by fileid to file data
				ClassFiles map[string]map[string]ICDiagnosisJobResponseFile `json:"class_files"`
			} `json:"inliers"`
		} `json:"datasets"`
	} `json:"tasks"`
}

// ImageMetadataRequest request type for image metadata
type ImageMetadataRequest struct {
	//the path to the image
	Path string `json:"path"`
}

// ImageMetadataResponse response type for image metadata
type ImageMetadataResponse struct {
	//the number of channels in an image
	Channels int `json:"channels"`
	//the height of the image
	Height int `json:"height"`
	//the width of the image
	Width int `json:"width"`
}
