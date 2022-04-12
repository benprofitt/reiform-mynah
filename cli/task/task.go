// Copyright (c) 2022 by Reiform. All Rights Reserved.

package task

import (
	"encoding/json"
	"errors"
	"fmt"
)

// MarshalJSON serializes the task
func (t *MynahTask) MarshalJSON() ([]byte, error) {
	return json.Marshal(t)
}

// UnmarshalJSON deserializes the task
func (t *MynahTask) UnmarshalJSON(bytes []byte) error {
	//check the task type
	var objMap map[string]*json.RawMessage

	if err := json.Unmarshal(bytes, &objMap); err != nil {
		return err
	}

	_, hasIdField := objMap["task_id"]
	_, hasTaskField := objMap["task_type"]
	_, hasDataField := objMap["task_data"]

	if hasIdField && hasTaskField && hasDataField {
		//deserialize the task id
		if err := json.Unmarshal(*objMap["task_id"], &t.TaskId); err != nil {
			return fmt.Errorf("error deserializing task_id: %s", err)
		}

		//deserialize the task type
		if err := json.Unmarshal(*objMap["task_type"], &t.TaskType); err != nil {
			return fmt.Errorf("error deserializing task_type: %s", err)
		}

		//look for the task struct type
		if taskStructFn, ok := taskConstructor[t.TaskType]; ok {
			taskStruct := taskStructFn()

			//unmarshal the actual task contents
			if err := json.Unmarshal(*objMap["task_data"], taskStruct); err != nil {
				return fmt.Errorf("error deserializing task_data: %s", err)
			}

			//set the task
			t.TaskData = taskStruct

			return nil

		} else {
			return fmt.Errorf("no mynah task with id %s", t.TaskType)
		}

	} else {
		return errors.New("mynah task missing 'task_id', 'task_type' or 'task_data'")
	}
}
