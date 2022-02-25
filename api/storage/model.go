// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"os"
	"reiform.com/mynah/model"
)

//Defines the interface the storage client must implement
type StorageProvider interface {
	//Save a file to the storage target. Local path passed to function, once function completes
	//file is moved to storage target
	StoreFile(*model.MynahFile, *model.MynahUser, func(*os.File) error) error
	//get the contents of a stored file. File is mounted locally, local path passed to function
	GetStoredFile(*model.MynahFile, func(*string) error) error
	//get the temporary path to a file
	GetTmpPath(*model.MynahFile) (string, error)
	//delete a stored file
	DeleteFile(*model.MynahFile) error
	//close the provider
	Close()
}
