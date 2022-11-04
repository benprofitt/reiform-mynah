// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"crypto/sha1" // #nosec
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
)

//local storage client implements StorageProvider
type localStorage struct {
	//the local path to store files
	localPath string
}

type savedLocalFile struct {
	// this version of the file
	version *model.MynahFileVersion
	// the path the file is stored at locally
	path string
}

// FileVersion gets the associated file
func (f *savedLocalFile) FileVersion() *model.MynahFileVersion {
	return f.version
}

// Path gets the local path to the file
func (f *savedLocalFile) Path() string {
	return f.path
}

// Close will clean the file up locally if stored elsewhere
func (f *savedLocalFile) Close() {
	// noop, local files won't be deleted
}

// return a path for this file by version id
func (s *localStorage) getVersionedPath(file *model.MynahFile, versionId model.MynahFileVersionId) string {
	return filepath.Join(s.localPath, fmt.Sprintf("%s_%s", file.Uuid, versionId))
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

// CopyFile copy the contents of a file to another with a different version id. Note: creates the new version id
func (s *localStorage) CopyFile(file *model.MynahFile, src model.MynahFileVersionId, dest model.MynahFileVersionId) error {
	//check that the source version id exists locally
	if version, ok := file.Versions[src]; !ok {
		return fmt.Errorf("source version id for file copy doesn't exist")
	} else if !version.ExistsLocally {
		return fmt.Errorf("source for file copy doesn't exist locally")
	}

	file.Versions[dest] = &model.MynahFileVersion{
		ExistsLocally: true,
		Metadata:      make(model.FileMetadata),
	}

	//copy metadata
	for k, v := range file.Versions[src].Metadata {
		file.Versions[dest].Metadata[k] = v
	}

	srcPath := s.getVersionedPath(file, src)

	//verify that the source file exists
	sourceFileStat, err := os.Stat(filepath.Clean(srcPath))
	if err != nil {
		return err
	}

	//check the mode
	if !sourceFileStat.Mode().IsRegular() {
		return fmt.Errorf("source file %s with version id %s is not a regular file", file.Uuid, src)
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

	destPath := s.getVersionedPath(file, dest)

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

// CreateTempFile creates a temp file. Note: must be cleaned up by the caller
func (s *localStorage) CreateTempFile(name string) (*os.File, error) {
	return os.Create(filepath.Clean(filepath.Join(s.localPath, name)))
}

// DeleteTempFile cleans up a temp file
func (s *localStorage) DeleteTempFile(name string) error {
	return os.Remove(filepath.Join(s.localPath, name))
}

// StoreFile Save a file to the storage target
func (s *localStorage) StoreFile(file *model.MynahFile, handler func(*os.File) error) error {
	//insert the default version id if not found
	if _, ok := file.Versions[model.OriginalVersionId]; !ok {
		file.Versions[model.OriginalVersionId] = &model.MynahFileVersion{
			ExistsLocally: true,
			Metadata:      make(model.FileMetadata),
		}
	}

	//create a local storage path for the file
	fullPath := s.getVersionedPath(file, model.OriginalVersionId)

	//create the local file to write to
	if localFile, err := os.Create(filepath.Clean(fullPath)); err == nil {

		handlerErr := handler(localFile)

		if handlerErr != nil {
			return handlerErr
		}

		//get the file size
		if stat, err := localFile.Stat(); err == nil {
			file.Versions[model.OriginalVersionId].Metadata[model.MetadataSize] = fmt.Sprintf("%d", stat.Size())

		} else {
			log.Warnf("failed to get filesize for %s: %s", file.Uuid, err)
		}

		//close the file before copying
		if err := localFile.Close(); err != nil {
			log.Errorf("error closing file %s: %s", file.Uuid, err)
		}

	} else {
		return err
	}

	//add the "latest" version id if not found
	if _, ok := file.Versions[model.LatestVersionId]; !ok {
		//copy the file
		return s.CopyFile(file, model.OriginalVersionId, model.LatestVersionId)

	} else {
		log.Infof("found latest version id when storing file, will not duplicate (if this is a new file, this is a bug)")
	}

	return nil
}

// GetStoredFile get the contents of a stored file
func (s *localStorage) GetStoredFile(file *model.MynahFile, versionId model.MynahFileVersionId, handler func(MynahLocalFile) error) error {
	if version, ok := file.Versions[versionId]; ok {
		if !version.ExistsLocally {
			return fmt.Errorf("file %s had a valid version id (%s) but is not available locally", file.Uuid, versionId)
		}

		//create the full temp path
		fullPath := s.getVersionedPath(file, versionId)

		//verify that the file exists
		_, err := os.Stat(fullPath)
		if err != nil {
			return err
		}

		return handler(&savedLocalFile{
			version: version,
			path:    fullPath,
		})

	} else {
		return fmt.Errorf("invalid version id for file %s: %s", file.Uuid, versionId)
	}
}

// GenerateSHA1Id takes the SHA1 of the latest version of the file
func (s *localStorage) GenerateSHA1Id(file *model.MynahFile) (model.MynahFileVersionId, error) {
	fullPath := s.getVersionedPath(file, model.LatestVersionId)

	//verify that the file exists
	_, err := os.Stat(fullPath)
	if err != nil {
		return "", err
	}

	fileReader, err := os.Open(filepath.Clean(fullPath))
	if err != nil {
		return "", err
	}

	hash := sha1.New() // #nosec
	if _, err := io.Copy(hash, fileReader); err != nil {
		log.Fatal(err)
	}
	sum := hash.Sum(nil)

	//create a new versionId from the hash
	return model.MynahFileVersionId(fmt.Sprintf("%x", sum)), nil
}

// DeleteFile delete a stored file
func (s *localStorage) DeleteFile(file *model.MynahFile) error {
	for versionId, version := range file.Versions {
		if version.ExistsLocally {
			if err := os.Remove(s.getVersionedPath(file, versionId)); err != nil {
				return fmt.Errorf("failed to delete file %s with versionId %s locally: %s",
					file.Uuid, versionId, err)
			}
		}
	}
	return nil
}

// Close the local storage provider (NOOP)
func (s *localStorage) Close() {
	log.Infof("local storage shutdown")
}
