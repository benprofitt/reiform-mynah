// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"fmt"
	"github.com/gabriel-vasile/mimetype"
	"io"
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

//check if this is an image mime type
func isImageType(mType *mimetype.MIME) bool {
	switch mType.String() {
	case "image/png":
		return true
	case "image/jpeg":
		return true
	case "image/tiff":
		return true
	default:
		return false
	}
}

// return a path for this file by tag
func (s *localStorage) getTaggedPath(file *model.MynahFile, tag model.MynahFileTag) string {
	return filepath.Join(s.localPath, fmt.Sprintf("%s_%s", file.Uuid, tag))
}

//copy the contents of a file to another with a different tag. Note: creates the new tag
func (s *localStorage) copyToTag(file *model.MynahFile, src model.MynahFileTag, dest model.MynahFileTag) error {
	//check that the source tag exists locally
	if version, ok := file.Versions[src]; !ok {
		return fmt.Errorf("source tag for file copy doesn't exist")
	} else if !version.ExistsLocally {
		return fmt.Errorf("source for file copy doesn't exist locally")
	}

	file.Versions[dest] = model.MynahFileVersion{
		ExistsLocally: true,
		Metadata:      make(model.FileMetadata),
	}

	//copy metadata
	for k, v := range file.Versions[src].Metadata {
		file.Versions[dest].Metadata[k] = v
	}

	srcPath := s.getTaggedPath(file, src)

	//verify that the source file exists
	sourceFileStat, err := os.Stat(filepath.Clean(srcPath))
	if err != nil {
		return err
	}

	//check the mode
	if !sourceFileStat.Mode().IsRegular() {
		return fmt.Errorf("source file %s with tag %s is not a regular file", file.Uuid, src)
	}

	source, err := os.Open(filepath.Clean(srcPath))
	if err != nil {
		return err
	}
	defer func(source *os.File) {
		err := source.Close()
		if err != nil {
			log.Warnf("failed to close file: %s", err)
		}
	}(source)

	destPath := s.getTaggedPath(file, dest)

	destination, err := os.Create(filepath.Clean(destPath))
	if err != nil {
		return err
	}
	defer func(destination *os.File) {
		err := destination.Close()
		if err != nil {
			log.Warnf("failed to close file: %s", err)
		}
	}(destination)

	_, err = io.Copy(destination, source)
	return err
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

// StoreFile Save a file to the storage target
func (s *localStorage) StoreFile(file *model.MynahFile, user *model.MynahUser, handler func(*os.File) error) error {
	//insert the default tag if not found
	if _, ok := file.Versions[model.TagOriginal]; !ok {
		file.Versions[model.TagOriginal] = model.MynahFileVersion{
			ExistsLocally: true,
			Metadata:      make(model.FileMetadata),
		}
	}

	//create a local storage path for the file
	fullPath := s.getTaggedPath(file, model.TagOriginal)

	//create the local file to write to
	if localFile, err := os.Create(filepath.Clean(fullPath)); err == nil {

		handlerErr := handler(localFile)

		if handlerErr != nil {
			return handlerErr
		}

		//get the file size
		if stat, err := localFile.Stat(); err == nil {
			file.Versions[model.TagOriginal].Metadata[model.MetadataSize] = fmt.Sprintf("%d", stat.Size())

		} else {
			log.Warnf("failed to get filesize for %s: %s", file.Uuid, err)
		}

		//check the mime type for image metadata
		if mimeType, err := mimetype.DetectFile(filepath.Clean(fullPath)); err == nil {

			if isImageType(mimeType) {
				//get metadata for image
				metadataRes, err := s.pyImplProvider.ImageMetadata(user, &pyimpl.ImageMetadataRequest{
					Path: fullPath,
				})

				if err == nil {
					file.Versions[model.TagOriginal].Metadata[model.MetadataWidth] = fmt.Sprintf("%d", metadataRes.Width)
					file.Versions[model.TagOriginal].Metadata[model.MetadataHeight] = fmt.Sprintf("%d", metadataRes.Height)
					file.Versions[model.TagOriginal].Metadata[model.MetadataChannels] = fmt.Sprintf("%d", metadataRes.Channels)
				} else {
					log.Warnf("failed to get metadata for %s: %s", file.Uuid, err)
				}
			} else {
				log.Infof("skipping metadata read for non image file type: %s", mimeType.String())
			}

		} else {
			log.Warnf("failed to read mime type for file %s: %s", file.Uuid, err)
		}

		//close the file before copying
		if err := localFile.Close(); err != nil {
			log.Errorf("error closing file %s: %s", file.Uuid, err)
		}

	} else {
		return err
	}

	//add the "latest" tag if not found
	if _, ok := file.Versions[model.TagLatest]; !ok {
		//copy the file
		return s.copyToTag(file, model.TagOriginal, model.TagLatest)

	} else {
		log.Infof("found latest tag when storing file, will not duplicate (if this is a new file, this is a bug)")
	}

	return nil
}

// GetStoredFile get the contents of a stored file
func (s *localStorage) GetStoredFile(file *model.MynahFile, tag model.MynahFileTag, handler func(*string) error) error {
	if location, ok := file.Versions[tag]; ok {
		if !location.ExistsLocally {
			return fmt.Errorf("file %s had a valid tag (%s) but is not available locally", file.Uuid, tag)
		}

		//create the full temp path
		fullPath := s.getTaggedPath(file, tag)

		//verify that the file exists
		_, err := os.Stat(fullPath)
		if err != nil {
			return err
		}
		return handler(&fullPath)
	} else {
		return fmt.Errorf("invalid tag for file %s: %s", file.Uuid, tag)
	}
}

// GetTmpPath get the temporary path to a file
func (s *localStorage) GetTmpPath(file *model.MynahFile, tag model.MynahFileTag) (string, error) {

	fullPath := s.getTaggedPath(file, tag)

	//verify that the file exists
	_, err := os.Stat(fullPath)
	if err != nil {
		return "", err
	}
	return fullPath, nil
}

// DeleteFile delete a stored file
func (s *localStorage) DeleteFile(file *model.MynahFile) error {
	var err error

	for tag, version := range file.Versions {
		if version.ExistsLocally {
			err = os.Remove(s.getTaggedPath(file, tag))
			if err != nil {
				log.Warnf("failed to delete file %s with tag %s locally: %s",
					file.Uuid, tag, err)
			}
		}

	}
	return err
}

// Close the local storage provider (NOOP)
func (s *localStorage) Close() {
	log.Infof("local storage shutdown")
}
