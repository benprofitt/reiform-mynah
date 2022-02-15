// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

//mynah python version response
type VersionResponse struct {
	Version string `json:"version"`
}

//format of files for a start diagnosis request
type DiagnosisJobFile struct {
	//the uuid for the file
	Uuid string `json:"uuid"`
	//the current class
	CurrentClass string `json:"current_class"`
	//the original class
	OriginalClass string `json:"original_class"`
	//projections
	//TODO
	ConfidenceVectors [][]float64 `json:"confidence_vectors"`
	//the type of the file
	MimeType string `json:"mime_type"`
}

//request format for a diagnosis job
type DiagnosisJobRequest struct {
	//all classes in the dataset
	Classes []string `json:"classes"`
	//map by class to map by fileid
	Files map[string]map[string]DiagnosisJobFile `json:"files"`
}

//response format for a diagnosis job
type DiagnosisJobResponse struct {
	//TODO
}
