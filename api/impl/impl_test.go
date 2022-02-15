// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

import (
	"fmt"
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/test"
	"testing"
)

//setup and teardown
func TestMain(m *testing.M) {
	dirPath := "data"

	//create the base directory if it doesn't exist
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		log.Fatalf("failed to create directory: %s", dirPath)
	}

	//run tests
	exitVal := m.Run()

	//remove generated
	if err := os.RemoveAll(dirPath); err != nil {
		log.Errorf("failed to clean up after tests: %s", err)
	}

	os.Exit(exitVal)
}

func TestMynahPythonVersion(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.ModulePath = "../../python"

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		if res, err := GetMynahImplVersion(c.PythonProvider); err == nil {
			if res.Version != "0.1.0" {
				return fmt.Errorf("unexpected version: %s", res.Version)
			}
			return nil
		} else {
			return err
		}
	})

	if err != nil {
		t.Fatalf("TestMynahPythonVersion error: %s", err)
	}
}
