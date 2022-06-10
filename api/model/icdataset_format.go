// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reiform.com/mynah/log"
)

// MynahICDatasetFolderFormat defines a basic dataset format,
// where classes are represented by directories
type MynahICDatasetFolderFormat struct{}

// create new task data structs by type identifier
var mynahIcDatasetFormatConstructor = map[MynahICDatasetFormatType]func() MynahICDatasetFormatMetadata{
	ICDatasetFolderFormat: func() MynahICDatasetFormatMetadata {
		return &MynahICDatasetFolderFormat{}
	},
}

// DatasetFileIterator takes a function that is called for each file to export
// the handler takes a file, the version to export, and the path within the zip archive to write to
func (f MynahICDatasetFolderFormat) DatasetFileIterator(version *MynahICDatasetVersion, fileNames map[MynahUuid]string, handler func(fileId MynahUuid, fileVersion MynahFileVersionId, filePath string) error) error {
	for fileId, fileData := range version.Files {
		fileName, ok := fileNames[fileId]
		if !ok {
			log.Warnf("no original filename provided for file %s when exporting in the simple folder format", fileId)
			fileName = string(fileId)
		}
		//this file will be saved in a folder with the name of its current class
		if err := handler(fileId, fileData.ImageVersionId, filepath.Join(string(fileData.CurrentClass), fileName)); err != nil {
			return fmt.Errorf("file export failed for %s", fileId)
		}
	}
	return nil
}

// GenerateArtifacts takes a handler that is called for each additional artifact the dataset export
// generates. THis includes the file contents and the path to write to in the zip archive
func (f MynahICDatasetFolderFormat) GenerateArtifacts(version *MynahICDatasetVersion, handler func(fileContents []byte, filePath string) error) error {
	//this format doesn't generate any additional artifacts
	return nil
}

// FromDB is used by xorm to deserialize
func (t *MynahICDatasetFormat) FromDB(bytes []byte) error {
	//check the type
	var objMap map[string]*json.RawMessage

	if err := json.Unmarshal(bytes, &objMap); err != nil {
		return err
	}

	_, hasTypeField := objMap["type"]
	_, hasMetadataField := objMap["metadata"]

	if hasTypeField && hasMetadataField {
		//deserialize the format identifier
		if err := json.Unmarshal(*objMap["type"], &t.Type); err != nil {
			return fmt.Errorf("error deserializing ic dataset format: %s", err)
		}

		//look for the task struct type
		if formatStructFn, ok := mynahIcDatasetFormatConstructor[t.Type]; ok {
			formatStruct := formatStructFn()

			//unmarshal the actual format contents
			if err := json.Unmarshal(*objMap["metadata"], formatStruct); err != nil {
				return fmt.Errorf("error deserializing ic dataset format metadata (type: %s): %s", t.Type, err)
			}

			//set the format metadata
			t.Metadata = formatStruct
			return nil

		} else {
			return fmt.Errorf("unknown ic dataset format task type: %s", t.Type)
		}

	} else {
		return errors.New("ic dataset format missing 'type' or 'metadata'")
	}
}

// ToDB is used by xorm to serialize
func (t MynahICDatasetFormat) ToDB() ([]byte, error) {
	return json.Marshal(t)
}
