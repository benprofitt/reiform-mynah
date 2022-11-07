// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"os"
	"reiform.com/mynah/model"
)

// MynahLocalFile represents a file stored locally
type MynahLocalFile interface {
	// FileVersion gets the associated file version data
	FileVersion() *model.MynahFileVersion
	// Path gets the local path to the file
	Path() string
}

// MynahLocalFileSet defines a set of files stored locally
type MynahLocalFileSet map[model.MynahUuid]MynahLocalFile

// StorageProvider Defines the interface the storage client must implement
type StorageProvider interface {
	// CopyFile copies a file from one version to another
	CopyFile(model.MynahUuid, *model.MynahFileVersion, *model.MynahFileVersion) error
	// CreateTempFile creates a temp file. Note: must be cleaned up by the caller with DeleteTempFile()
	CreateTempFile(string) (*os.File, error)
	// DeleteTempFile cleans up a temp file
	DeleteTempFile(string) error
	// StoreFile Save a _new_ file to the storage target. Local path passed to function, once function completes
	//file is moved to storage target
	StoreFile(*model.MynahFile, func(*os.File) error) error
	// GetStoredFile get the contents of a stored file. File is mounted locally, local path passed to function
	GetStoredFile(model.MynahUuid, *model.MynahFileVersion, func(MynahLocalFile) error) error
	// GetStoredFiles get the contents of some set of files. Files are mounted locally, local paths passed to function
	GetStoredFiles(model.MynahVersionedFileSet, func(MynahLocalFileSet) error) error
	// GenerateSHA1Id takes the SHA1 of some version of the file
	GenerateSHA1Id(model.MynahUuid, *model.MynahFileVersion) (model.MynahFileVersionId, error)
	// DeleteFileVersion delete a stored file (by version)
	DeleteFileVersion(model.MynahUuid, *model.MynahFileVersion) error
	// DeleteAllFileVersions deletes all versions of a file
	DeleteAllFileVersions(*model.MynahFile) error
	// Close the provider
	Close()
}
