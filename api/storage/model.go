package storage

import (
  "reiform.com/mynah/model"
  "os"
)

//Defines the interface the storage client must implement
type StorageProvider interface {
  //Save a file to the storage target. Local path passed to function, once function completes
  //file is moved to storage target
  StoreFile(*model.MynahFile, func (*os.File) error) error
  //get the contents of a stored file. File is mounted locally, local path passed to function
  GetStoredFile(*model.MynahFile, func (*string) error) error
  //delete a stored file
  DeleteFile(*model.MynahFile) error
}

//local storage client adheres to StorageProvider
type localStorage struct {
  //the local path to store files
  localPath string
}

//external storage client adheres to StorageProvider
type externalStorage struct {
}
