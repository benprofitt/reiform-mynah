// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"os"
	"reiform.com/mynah/model"
)

// StorageProvider Defines the interface the storage client must implement
type StorageProvider interface {
	// CopyFile copies a file from one version to another
	CopyFile(*model.MynahFile, model.MynahFileVersionId, model.MynahFileVersionId) error
	// StoreFile Save a file to the storage target. Local path passed to function, once function completes
	//file is moved to storage target
	StoreFile(*model.MynahFile, *model.MynahUser, func(*os.File) error) error
	// GetStoredFile get the contents of a stored file. File is mounted locally, local path passed to function
	GetStoredFile(*model.MynahFile, model.MynahFileVersionId, func(*string) error) error
	// GetTmpPath get the temporary path to a file
	GetTmpPath(*model.MynahFile, model.MynahFileVersionId) (string, error)
	// GenerateSHA1Id takes the SHA1 of the latest version of the file
	GenerateSHA1Id(*model.MynahFile) (model.MynahFileVersionId, error)
	// DeleteFile delete a stored file
	DeleteFile(*model.MynahFile) error
	// Close close the provider
	Close()
}
