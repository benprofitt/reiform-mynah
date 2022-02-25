// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"reiform.com/mynah/model"
)

type PyImplProvider interface {
	//get the current version
	GetMynahImplVersion() (*VersionResponse, error)
	//start ic diagnosis
	ICDiagnosisJob(*model.MynahUser, *ICDiagnosisJobRequest) (*ICDiagnosisJobResponse, error)
	//get image metadata
	ImageMetadata(*model.MynahUser, *ImageMetadataRequest) (*ImageMetadataResponse, error)
}

//mynah python version response
type VersionResponse struct {
	//the version returned by python
	Version string `json:"version"`
}

//format of files for a start diagnosis request
type ICDiagnosisJobFile struct {
	//the uuid for the file
	Uuid string `json:"uuid"`
	//the current class
	CurrentClass string `json:"current_class"`
	//the original class
	OriginalClass string `json:"original_class"`
	//projections
	//TODO
	ConfidenceVectors [][]float64 `json:"confidence_vectors"`
	//the path to the file
	TmpPath string `json:"tmp_path"`
}

//request format for a diagnosis job
type ICDiagnosisJobRequest struct {
	//all classes in the dataset
	Classes []string `json:"classes"`
	//map by class to map by fileid
	ClassFiles map[string]map[string]ICDiagnosisJobFile `json:"class_files"`
}

//response format for a diagnosis job
type ICDiagnosisJobResponse struct {
	//TODO
}

//request type for image metadata
type ImageMetadataRequest struct {
	//the path to the image
	Path string `json:"path"`
}

//response type for image metadata
type ImageMetadataResponse struct {
	//the number of channels in an image
	Channels int `json:"channels"`
	//the height of the image
	Height int `json:"height"`
	//the width of the image
	Width int `json:"width"`
}
