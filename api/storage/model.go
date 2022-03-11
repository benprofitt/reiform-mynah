// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"os"
	"reiform.com/mynah/model"
)

// StorageProvider Defines the interface the storage client must implement
type StorageProvider interface {
	// StoreFile Save a file to the storage target. Local path passed to function, once function completes
	//file is moved to storage target
	StoreFile(*model.MynahFile, *model.MynahUser, func(*os.File) error) error
	// GetStoredFile get the contents of a stored file. File is mounted locally, local path passed to function
	GetStoredFile(*model.MynahFile, model.MynahFileTag, func(*string) error) error
	// GetTmpPath get the temporary path to a file
	GetTmpPath(*model.MynahFile, model.MynahFileTag) (string, error)
	// DeleteFile delete a stored file
	DeleteFile(*model.MynahFile) error
	// Close close the provider
	Close()
}
