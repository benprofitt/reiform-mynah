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
	// Close will clean the file up locally if stored elsewhere
	Close()
}

// StorageProvider Defines the interface the storage client must implement
type StorageProvider interface {
	// CopyFile copies a file from one version to another
	CopyFile(*model.MynahFile, model.MynahFileVersionId, model.MynahFileVersionId) error
	// CreateTempFile creates a temp file. Note: must be cleaned up by the caller with DeleteTempFile()
	CreateTempFile(string) (*os.File, error)
	// DeleteTempFile cleans up a temp file
	DeleteTempFile(string) error
	// StoreFile Save a file to the storage target. Local path passed to function, once function completes
	//file is moved to storage target
	StoreFile(*model.MynahFile, func(*os.File) error) error
	// GetStoredFile get the contents of a stored file. File is mounted locally, local path passed to function
	GetStoredFile(*model.MynahFile, model.MynahFileVersionId, func(MynahLocalFile) error) error
	// GenerateSHA1Id takes the SHA1 of the latest version of the file
	GenerateSHA1Id(*model.MynahFile) (model.MynahFileVersionId, error)
	// DeleteFile delete a stored file
	DeleteFile(*model.MynahFile) error
	// Close the provider
	Close()
}
