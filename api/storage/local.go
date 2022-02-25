// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"fmt"
	"os"
	"path/filepath"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/pyimpl"
	"reiform.com/mynah/settings"
)

//local storage client implements StorageProvider
type localStorage struct {
	//the local path to store files
	localPath string
	//the python interface provider
	pyImplProvider pyimpl.PyImplProvider
}

//create a new local storage provider
func newLocalStorage(mynahSettings *settings.MynahSettings, pyImplProvider pyimpl.PyImplProvider) (*localStorage, error) {
	//create the storage directory if it doesn't exist
	if err := os.MkdirAll(mynahSettings.StorageSettings.LocalPath, os.ModePerm); err != nil {
		return nil, err
	}
	return &localStorage{
		localPath:      mynahSettings.StorageSettings.LocalPath,
		pyImplProvider: pyImplProvider,
	}, nil
}

//Save a file to the storage target
func (s *localStorage) StoreFile(file *model.MynahFile, user *model.MynahUser, handler func(*os.File) error) error {
	//create a local storage path for the file
	fullPath := filepath.Join(s.localPath, file.Uuid)

	//create the local file to write to
	if localFile, err := os.Create(filepath.Clean(fullPath)); err == nil {
		defer func() {
			if err := localFile.Close(); err != nil {
				log.Errorf("error closing file %s: %s", file.Uuid, err)
			}
		}()
		handlerErr := handler(localFile)

		if handlerErr != nil {
			//get the file size
			if stat, err := localFile.Stat(); err == nil {
				file.Metadata[model.MetadataSize] = fmt.Sprintf("%d", stat.Size())

			} else {
				log.Warnf("failed to get filesize for %s: %s", file.Uuid, err)
			}

			//get metadata for image
			metadataRes, err := s.pyImplProvider.ImageMetadata(user, &pyimpl.ImageMetadataRequest{
				Path: fullPath,
			})

			if err == nil {
				file.Metadata[model.MetadataWidth] = fmt.Sprintf("%d", metadataRes.Width)
				file.Metadata[model.MetadataHeight] = fmt.Sprintf("%d", metadataRes.Height)
				file.Metadata[model.MetadataChannels] = fmt.Sprintf("%d", metadataRes.Channels)
			} else {
				log.Warnf("failed to get metadata for %s: %s", file.Uuid, err)
			}
		}

		return handlerErr
	} else {
		return err
	}
}

//get the contents of a stored file
func (s *localStorage) GetStoredFile(file *model.MynahFile, handler func(*string) error) error {
	fullPath := filepath.Join(s.localPath, file.Uuid)

	//verify that the file exists
	_, err := os.Stat(fullPath)
	if err != nil {
		return err
	}
	return handler(&fullPath)
}

//get the temporary path to a file
func (s *localStorage) GetTmpPath(file *model.MynahFile) (string, error) {
	fullPath := filepath.Join(s.localPath, file.Uuid)

	//verify that the file exists
	_, err := os.Stat(fullPath)
	if err != nil {
		return "", err
	}
	return fullPath, nil
}

//delete a stored file
func (s *localStorage) DeleteFile(file *model.MynahFile) error {
	return os.Remove(filepath.Join(s.localPath, file.Uuid))
}

func (s *localStorage) Close() {
	log.Infof("local storage shutdown")
}
