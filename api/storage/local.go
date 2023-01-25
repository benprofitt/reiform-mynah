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

// return a path for this file by version id
func (s *localStorage) getVersionedPath(id model.MynahUuid, versionId model.MynahFileVersionId) string {
	return filepath.Join(s.localPath, fmt.Sprintf("%s_%s", id, versionId))
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
func (s *localStorage) CopyFile(fileId model.MynahUuid, src, dest *model.MynahFileVersion) error {
	src.CopyTo(dest)

	srcPath := s.getVersionedPath(fileId, src.VersionId)

	//verify that the source file exists
	sourceFileStat, err := os.Stat(filepath.Clean(srcPath))
	if err != nil {
		return err
	}

	//check the mode
	if !sourceFileStat.Mode().IsRegular() {
		return fmt.Errorf("source file %s with version id %s is not a regular file", fileId, src.VersionId)
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

	destPath := s.getVersionedPath(fileId, dest.VersionId)

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
	// set original and latest versions
	file.Versions[model.OriginalVersionId] = model.NewMynahFileVersion(model.OriginalVersionId)
	file.Versions[model.LatestVersionId] = model.NewMynahFileVersion(model.LatestVersionId)
	file.Versions[model.OriginalVersionId].ExistsLocally = true
	file.Versions[model.LatestVersionId].ExistsLocally = true

	//create a local storage path for the file
	fullPath := s.getVersionedPath(file.Uuid, model.OriginalVersionId)

	//create the local file to write to
	if localFile, err := os.Create(filepath.Clean(fullPath)); err == nil {

		handlerErr := handler(localFile)

		if handlerErr != nil {
			return handlerErr
		}

		//get the file size
		if stat, err := localFile.Stat(); err == nil {
			file.Versions[model.OriginalVersionId].Metadata.Size = stat.Size()

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

	//copy the file to the latest version
	return s.CopyFile(file.Uuid, file.Versions[model.OriginalVersionId], file.Versions[model.LatestVersionId])
}

// GetStoredFile get the contents of a stored file
func (s *localStorage) GetStoredFile(fileId model.MynahUuid, version *model.MynahFileVersion, handler func(MynahLocalFile) error) error {
	if !version.ExistsLocally {
		return fmt.Errorf("file %s had a valid version id (%s) but is not available locally", fileId, version.VersionId)
	}

	//create the full temp path
	fullPath := s.getVersionedPath(fileId, version.VersionId)

	//verify that the file exists
	_, err := os.Stat(fullPath)
	if err != nil {
		return err
	}

	return handler(&savedLocalFile{
		version: version,
		path:    fullPath,
	})
}

// GetStoredFiles get the contents of some set of files. Files are mounted locally, local paths passed to function
func (s *localStorage) GetStoredFiles(fileVersions model.MynahVersionedFileSet, handler func(MynahLocalFileSet) error) error {
	localFiles := make(MynahLocalFileSet)

	for fileId, version := range fileVersions {
		if !version.ExistsLocally {
			return fmt.Errorf("file %s had a valid version id (%s) but is not available locally", fileId, version.VersionId)
		}

		//create the full temp path
		fullPath := s.getVersionedPath(fileId, version.VersionId)

		//verify that the file exists
		_, err := os.Stat(fullPath)
		if err != nil {
			return err
		}

		localFiles[fileId] = &savedLocalFile{
			version: version,
			path:    fullPath,
		}
	}

	return handler(localFiles)
}

// GenerateSHA1Id takes the SHA1 of some version of the file
func (s *localStorage) GenerateSHA1Id(fileId model.MynahUuid, version *model.MynahFileVersion) (model.MynahFileVersionId, error) {
	fullPath := s.getVersionedPath(fileId, version.VersionId)

	//verify that the file exists
	_, err := os.Stat(fullPath)
	if err != nil {
		return "", err
	}

	fileReader, err := os.Open(filepath.Clean(fullPath))
	if err != nil {
		return "", err
	}
	defer func(f *os.File) {
		err := f.Close()
		if err != nil {
			log.Warnf("failed to close file: %s", err)
		}
	}(fileReader)

	hash := sha1.New() // #nosec
	if _, err := io.Copy(hash, fileReader); err != nil {
		log.Fatal(err)
	}
	sum := hash.Sum(nil)

	//create a new versionId from the hash
	return model.MynahFileVersionId(fmt.Sprintf("%x", sum)), nil
}

// DeleteFileVersion delete a stored file
func (s *localStorage) DeleteFileVersion(fileId model.MynahUuid, version *model.MynahFileVersion) error {
	if version.ExistsLocally {
		if err := os.Remove(s.getVersionedPath(fileId, version.VersionId)); err != nil {
			return fmt.Errorf("failed to delete file %s with versionId %s locally: %s",
				fileId, version.VersionId, err)
		}
	}
	return nil
}

// DeleteAllFileVersions deletes all versions of a file
func (s *localStorage) DeleteAllFileVersions(file *model.MynahFile) error {
	for _, version := range file.Versions {
		if err := s.DeleteFileVersion(file.Uuid, version); err != nil {
			return err
		}
	}
	return nil
}

// Close the local storage provider (NOOP)
func (s *localStorage) Close() {
	log.Infof("local storage shutdown")
}
