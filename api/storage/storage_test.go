// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"errors"
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
	if storagePErr != nil {
		t.Errorf("failed to create storage provider for test %s", storagePErr)
		return
	}
	defer storageProvider.Close()

	file := model.MynahFile{
		Uuid:     "mynah_test_file",
		Location: model.Local,
	}

	//store the file
	if storeErr := storageProvider.StoreFile(&file, func(f *os.File) error {
		//write to the file
		_, err := f.WriteString("data")
		return err

	}); storeErr != nil {
		t.Errorf("failed to store file %s", storeErr)
		return
	}

	//get the stored file
	if getErr := storageProvider.GetStoredFile(&file, func(p *string) error {
		if *p != "data/tmp/mynah_test_file" {
			return errors.New("test file does not exist")
		}
		//check that the file exists
		_, err := os.Stat(*p)
		return err

	}); getErr != nil {
		t.Errorf("failed to get stored file %s", getErr)
		return
	}

	//delete the file
	if deleteErr := storageProvider.DeleteFile(&file); deleteErr == nil {
		//verify that the file doesn't exist
		if _, err := os.Stat("data/tmp/mynah_test_file"); err == nil {
			t.Error("file was not deleted successfully")
			return
		}

		//verify that get file returns an error
		getErr := storageProvider.GetStoredFile(&file, func(p *string) error { return nil })
		if getErr == nil {
			t.Error("get stored file did not return an error after file was deleted")
			return
		}

		//verify that a second deletion returns an error
		if deleteErr2 := storageProvider.DeleteFile(&file); deleteErr2 == nil {
			t.Error("second file deletion did not return an error")
			return
		}

	} else {
		t.Errorf("failed to delete file %s", deleteErr)
		return
	}
}
