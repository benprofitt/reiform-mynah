// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"os"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/mynahExec"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"testing"
)

// withPyImplProvider initialize a python impl provider for tests
func withPyImplProvider(mynahSettings *settings.MynahSettings, handler func(provider ImplProvider) error) error {
	storageProvider, storagePErr := storage.NewStorageProvider(mynahSettings)
	if storagePErr != nil {
		return fmt.Errorf("failed to create storage provider for test %s", storagePErr)
	}
	defer storageProvider.Close()

	authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	if authErr != nil {
		return fmt.Errorf("failed to create auth provider for test %s", authErr)
	}
	defer authProvider.Close()

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	if dbErr != nil {
		return fmt.Errorf("failed to create db provider for test %s", authErr)
	}
	defer dbProvider.Close()

	executor, err := mynahExec.NewLocalExecutor(mynahSettings)
	if err != nil {
		return fmt.Errorf("failed to create executor for test %s", authErr)
	}
	defer executor.Close()

	return handler(NewImplProvider(mynahSettings, dbProvider, storageProvider, executor))
}

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

// TestMynahPythonVersion verifies that the python version is the same as the app version
func TestMynahPythonVersion(t *testing.T) {
	s := settings.DefaultSettings()
	s.PythonSettings.PythonExecutable = "../../python/python_version_test.py"

	err := withPyImplProvider(s, func(provider ImplProvider) error {
		res, err := provider.GetMynahImplVersion()
		require.NoError(t, err)
		require.Equal(t, settings.MynahApplicationVersion, res.Version)
		return nil
	})

	require.NoError(t, err)
}
