// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"errors"
	"fmt"
	"github.com/stretchr/testify/require"
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
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

//TODO run tests for both storage providers by changing the settings and passing in

//Test basic storage behavior
func TestBasicStorageActions(t *testing.T) {
	s := settings.DefaultSettings()

	storageProvider, storagePErr := NewStorageProvider(s)
	require.NoError(t, storagePErr)
	defer storageProvider.Close()

	file := model.MynahFile{
		Uuid:     "mynah_test_file",
		Versions: make(map[model.MynahFileVersionId]*model.MynahFileVersion),
	}

	//store the file
	require.NoError(t, storageProvider.StoreFile(&file, func(f *os.File) error {
		//write to the file
		_, err := f.WriteString("data")
		return err
	}))

	expectedPath := fmt.Sprintf("data/tmp/%s_%s", file.Uuid, model.LatestVersionId)

	//get the stored file
	require.NoError(t, storageProvider.GetStoredFile(file.Uuid, file.Versions[model.LatestVersionId], func(lf MynahLocalFile) error {
		if lf.Path() != expectedPath {
			return errors.New("integration file does not exist")
		}
		//check that the file exists
		_, err := os.Stat(lf.Path())
		return err
	}))

	require.NoError(t, storageProvider.DeleteAllFileVersions(&file))

	_, err := os.Stat(expectedPath)
	require.Error(t, err)

	require.Error(t, storageProvider.GetStoredFile(file.Uuid, file.Versions[model.LatestVersionId], func(MynahLocalFile) error { return nil }))
	require.Error(t, storageProvider.DeleteAllFileVersions(&file))
}
