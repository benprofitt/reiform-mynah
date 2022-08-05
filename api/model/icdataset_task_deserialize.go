// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
	"encoding/json"
	"errors"
	"fmt"
)

// create new task data structs by type identifier
var mynahICProcessTaskConstructor = map[MynahICProcessTaskType]func() MynahICProcessTaskMetadata{
	ICProcessDiagnoseMislabeledImagesTask: func() MynahICProcessTaskMetadata {
		return &MynahICProcessTaskDiagnoseMislabeledImagesMetadata{}
	},
	ICProcessCorrectMislabeledImagesTask: func() MynahICProcessTaskMetadata {
		return &MynahICProcessTaskCorrectMislabeledImagesMetadata{}
	},
	ICProcessDiagnoseClassSplittingTask: func() MynahICProcessTaskMetadata {
		return &MynahICProcessTaskDiagnoseClassSplittingMetadata{}
	},
	ICProcessCorrectClassSplittingTask: func() MynahICProcessTaskMetadata {
		return &MynahICProcessTaskCorrectClassSplittingMetadata{}
	},
}

// create new task data structs by type identifier
var mynahICProcessDiagnosisTasks = map[MynahICProcessTaskType]interface{}{
	ICProcessDiagnoseMislabeledImagesTask: nil,
	ICProcessDiagnoseClassSplittingTask:   nil,
}

// IsMynahICDiagnosisTask returns true if the task type corresponds to a diagnosis process
func IsMynahICDiagnosisTask(taskType MynahICProcessTaskType) bool {
	_, ok := mynahICProcessDiagnosisTasks[taskType]
	return ok
}

// map from task type to report structure
var mynahICProcessTaskReportConstructor = map[MynahICProcessTaskType]func() MynahICProcessTaskReportMetadata{
	ICProcessDiagnoseMislabeledImagesTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskDiagnoseMislabeledImagesReport{}
	},
	ICProcessCorrectMislabeledImagesTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskCorrectMislabeledImagesReport{}
	},
	ICProcessDiagnoseClassSplittingTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskDiagnoseClassSplittingReport{}
	},
	ICProcessCorrectClassSplittingTask: func() MynahICProcessTaskReportMetadata {
		return &MynahICProcessTaskCorrectClassSplittingReport{}
	},
}

// ValidMynahICProcessTaskType verifies that the task identifier is known
func ValidMynahICProcessTaskType(val MynahICProcessTaskType) bool {
	_, tOk := mynahICProcessTaskConstructor[val]
	_, rOk := mynahICProcessTaskReportConstructor[val]
	return tOk && rOk
}

// UnmarshalJSON deserializes the task metadata
func (t *MynahICProcessTaskData) UnmarshalJSON(bytes []byte) error {
	//check the task type
	var objMap map[string]*json.RawMessage

	if err := json.Unmarshal(bytes, &objMap); err != nil {
		return err
	}

	_, hasTypeField := objMap["type"]
	_, hasMetadataField := objMap["metadata"]

	if hasTypeField && hasMetadataField {
		//deserialize the task id
		if err := json.Unmarshal(*objMap["type"], &t.Type); err != nil {
			return fmt.Errorf("error deserializing ic process task: %s", err)
		}

		//look for the task struct type
		if taskStructFn, ok := mynahICProcessTaskConstructor[t.Type]; ok {
			taskStruct := taskStructFn()

			//unmarshal the actual task contents
			if err := json.Unmarshal(*objMap["metadata"], taskStruct); err != nil {
				return fmt.Errorf("error deserializing ic process task metadata (type: %s): %s", t.Type, err)
			}

			//set the task
			t.Metadata = taskStruct
			return nil

		} else {
			return fmt.Errorf("unknown ic process task type: %s", t.Type)
		}

	} else {
		return errors.New("ic process task missing 'type' or 'metadata'")
	}
}

// UnmarshalJSON deserializes the report metadata
func (t *MynahICProcessTaskReportData) UnmarshalJSON(bytes []byte) error {
	//check the task type
	var objMap map[string]*json.RawMessage

	if err := json.Unmarshal(bytes, &objMap); err != nil {
		return err
	}

	_, hasTypeField := objMap["type"]
	_, hasMetadataField := objMap["metadata"]

	if hasTypeField && hasMetadataField {
		//deserialize the task id
		if err := json.Unmarshal(*objMap["type"], &t.Type); err != nil {
			return fmt.Errorf("error deserializing ic process report task: %s", err)
		}

		//create the task struct
		if taskStructFn, ok := mynahICProcessTaskReportConstructor[t.Type]; ok {
			taskStruct := taskStructFn()
			//unmarshal the actual task contents
			if err := json.Unmarshal(*objMap["metadata"], taskStruct); err != nil {
				return fmt.Errorf("error deserializing ic process task report (type: %s): %s", t.Type, err)
			}

			//set the task data
			t.Metadata = taskStruct
			return nil
		} else {
			return fmt.Errorf("unknown ic process task type: %s", t.Type)
		}

	} else {
		return errors.New("ic process task report missing 'type' or 'metadata'")
	}
}
