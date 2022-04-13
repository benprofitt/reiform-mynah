// Copyright (c) 2022 by Reiform. All Rights Reserved.

package task

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
	"reiform.com/mynah-cli/server"
	"reiform.com/mynah/api"
	"reiform.com/mynah/model"
)

// CreatedICDatasetKey is the key for the create dataset result context
const CreatedICDatasetKey contextKey = "CreatedICDataset"

// MynahCreateICDatasetTask defines the task of creating an image classification dataset
type MynahCreateICDatasetTask struct {
	//reference files by id already in mynah (map fileid to class name)
	FromExisting map[string]string `json:"files"`
	//reference files uploaded by previous tasks
	FromTasks []MynahTaskId `json:"from_tasks"`
	//regex to assign a file a class name based on path
	LocalPathClassnameRegex string `json:"local_path_classname_regex"`
	//the name for the dataset
	DatasetName string `json:"dataset_name"`
}

//assign a class name for this file
func assignClassName(regex *regexp.Regexp, localPath string) (string, error) {
	if regex != nil {
		//attempt to match
		if res := regex.FindStringSubmatch(localPath); len(res) > 0 {
			return res[len(res)-1], nil
		} else {
			return "", fmt.Errorf("class name regex found no matches for %s", localPath)
		}
	} else {
		return filepath.Base(filepath.Dir(localPath)), nil
	}
}

// ExecuteTask executes the create icdataset task
func (t MynahCreateICDatasetTask) ExecuteTask(mynahServer *server.MynahClient,
	tctx MynahTaskContext) (context.Context, error) {

	if t.FromExisting == nil {
		t.FromExisting = make(map[string]string)
	}

	var classRegex *regexp.Regexp

	//create regex is applicable
	if len(t.LocalPathClassnameRegex) > 0 {
		if r, err := regexp.Compile(t.LocalPathClassnameRegex); err == nil {
			classRegex = r
		} else {
			return nil, errors.New("failed to compile regex for file class assignment")
		}
	}

	for _, uploadTask := range t.FromTasks {
		//get the set from context
		if set, ok := tctx[uploadTask]; ok {
			if uploadedSet, ok := set.Value(UploadedFilesSetKey).(UploadedFilesSet); ok {
				for fileid, path := range uploadedSet {
					//generate the class name
					if className, err := assignClassName(classRegex, path); err == nil {
						t.FromExisting[fileid] = className
					} else {
						return nil, fmt.Errorf("failed to assign class name for file: %s", err)
					}
				}
			} else {
				return nil, fmt.Errorf("task %s does not have uploaded files to add to dataset", uploadTask)
			}

		} else {
			return nil, fmt.Errorf("no such task: %s", uploadTask)
		}
	}

	datasetRequest := api.CreateICDatasetRequest{
		Name:  t.DatasetName,
		Files: t.FromExisting,
	}

	var icDatasetResponse model.MynahICDataset

	//make the request
	if err := mynahServer.ExecutePostJsonRequest("icdataset/create", &datasetRequest, &icDatasetResponse); err != nil {
		return nil, fmt.Errorf("failed to create ic dataset: %s", err)
	}

	//add the id to the context
	return context.WithValue(context.Background(), CreatedICDatasetKey, icDatasetResponse.Uuid), nil
}
