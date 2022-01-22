package storage

import (
	"errors"
	"os"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"testing"
)

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
		if *p != "tmp/mynah_test_file" {
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
		if _, err := os.Stat("tmp/mynah_test_file"); err == nil {
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
