// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
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

	//initialize python
	pythonProvider := python.NewPythonProvider(mynahSettings)
	defer pythonProvider.Close()

	//create the python impl provider
	pyImplProvider := NewPyImplProvider(mynahSettings, pythonProvider)

	if res, err := pyImplProvider.GetMynahImplVersion(); err == nil {
		if res.Version != "0.1.0" {
			t.Fatalf("unexpected version: %s", res.Version)
		}
	} else {
		t.Fatalf("error calling python: %s", err)
	}
}
