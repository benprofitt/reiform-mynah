// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"errors"
	"fmt"
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/pyimpl"
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

//TODO run tests for both storage providers by changing the settings and passing in

//Test basic storage behavior
func TestBasicStorageActions(t *testing.T) {
	s := settings.DefaultSettings()

	//initialize python
	pythonProvider := python.NewPythonProvider(s)
	defer pythonProvider.Close()

	//create the python impl provider
	pyImplProvider := pyimpl.NewPyImplProvider(s, pythonProvider)

	storageProvider, storagePErr := NewStorageProvider(s, pyImplProvider)
	if storagePErr != nil {
		t.Errorf("failed to create storage provider for test %s", storagePErr)
		return
	}
	defer storageProvider.Close()

	file := model.MynahFile{
		Uuid:     "mynah_test_file",
		Versions: make(map[model.MynahFileTag]*model.MynahFileVersion),
	}

	user := model.MynahUser{
		Uuid: "owner",
	}

	//store the file
	if storeErr := storageProvider.StoreFile(&file, &user, func(f *os.File) error {
		//write to the file
		_, err := f.WriteString("data")
		return err

	}); storeErr != nil {
		t.Errorf("failed to store file %s", storeErr)
		return
	}

	expectedPath := fmt.Sprintf("data/tmp/%s_%s", file.Uuid, model.TagLatest)

	//get the stored file
	if getErr := storageProvider.GetStoredFile(&file, model.TagLatest, func(p *string) error {
		if *p != expectedPath {
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
		if _, err := os.Stat(expectedPath); err == nil {
			t.Error("file was not deleted successfully")
			return
		}

		//verify that get file returns an error
		getErr := storageProvider.GetStoredFile(&file, model.TagLatest, func(p *string) error { return nil })
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

//test size detection
func TestStorageImageSizeDetection(t *testing.T) {
	s := settings.DefaultSettings()
	s.PythonSettings.ModulePath = "../../python"

	//initialize python
	pythonProvider := python.NewPythonProvider(s)
	defer pythonProvider.Close()

	//create the python impl provider
	pyImplProvider := pyimpl.NewPyImplProvider(s, pythonProvider)

	storageProvider, storagePErr := NewStorageProvider(s, pyImplProvider)
	if storagePErr != nil {
		t.Errorf("failed to create storage provider for test %s", storagePErr)
		return
	}
	defer storageProvider.Close()

	jpegPath := "../../docs/test_image.jpg"
	pngPath := "../../docs/mynah_arch_1-13-21.drawio.png"
	notImagePath := "../../docs/ipc.md"

	user := model.MynahUser{
		Uuid: "owner",
	}

	//get metadata for image
	jpegRes, err := pyImplProvider.ImageMetadata(&user, &pyimpl.ImageMetadataRequest{
		Path: jpegPath,
	})

	if err != nil {
		t.Fatalf("error requesting metadata: %s", err)
	}

	if (jpegRes.Width != 4032) || (jpegRes.Height != 3024) {
		t.Fatalf("unexpected jpeg dimensions (%d, %d), expected: (%d, %d)",
			jpegRes.Width, jpegRes.Height, 4032, 3024)
	}

	pngRes, err := pyImplProvider.ImageMetadata(&user, &pyimpl.ImageMetadataRequest{
		Path: pngPath,
	})

	if err != nil {
		t.Fatalf("error requesting metadata: %s", err)
	}

	if (pngRes.Width != 3429) || (pngRes.Height != 2316) {
		t.Fatalf("unexpected png dimensions (%d, %d), expected: (%d, %d)",
			pngRes.Width, pngRes.Height, 3429, 2316)
	}

	_, err = pyImplProvider.ImageMetadata(&user, &pyimpl.ImageMetadataRequest{
		Path: notImagePath,
	})

	if err == nil {
		t.Fatalf("expected error for non-image")
	}
}
