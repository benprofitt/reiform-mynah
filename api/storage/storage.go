package storage

import (
	"reiform.com/mynah/settings"
)

//Create a new storage provider based on the Mynah settings
func NewStorageProvider(mynahSettings *settings.MynahSettings) (StorageProvider, error) {
	//for now just return local provider
	return newLocalStorage(mynahSettings)
}
