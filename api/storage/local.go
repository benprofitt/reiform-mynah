package storage

import (
	"log"
	"os"
	"path/filepath"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
)

//local storage client implements StorageProvider
type localStorage struct {
	//the local path to store files
	localPath string
}

//create a new local storage provider
func newLocalStorage(mynahSettings *settings.MynahSettings) (*localStorage, error) {
	//create the storage directory if it doesn't exist
	if err := os.MkdirAll(mynahSettings.StorageSettings.LocalPath, os.ModePerm); err != nil {
		return nil, err
	}
	return &localStorage{
		localPath: mynahSettings.StorageSettings.LocalPath,
	}, nil
}

//Save a file to the storage target
func (s *localStorage) StoreFile(file *model.MynahFile, handler func(*os.File) error) error {
	//create a local storage path for the file
	file.Path = filepath.Join(s.localPath, file.Uuid)
	file.Location = model.Local

	//create the local file to write to
	if localFile, err := os.Create(file.Path); err == nil {
		defer localFile.Close()
		return handler(localFile)
	} else {
		return err
	}
}

//get the contents of a stored file
func (s *localStorage) GetStoredFile(file *model.MynahFile, handler func(*string) error) error {
	//verify that the file exists
	_, err := os.Stat(file.Path)
	if err != nil {
		return err
	}
	return handler(&file.Path)
}

//delete a stored file
func (s *localStorage) DeleteFile(file *model.MynahFile) error {
	return os.Remove(file.Path)
}

func (s *localStorage) Close() {
	log.Printf("local storage shutdown")
}
