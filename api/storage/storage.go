// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"reiform.com/mynah/pyimpl"
	"reiform.com/mynah/settings"
)

// NewStorageProvider Create a new storage provider based on the Mynah settings
func NewStorageProvider(mynahSettings *settings.MynahSettings, pyimplProvider pyimpl.PyImplProvider) (StorageProvider, error) {
	//for now just return local provider
	return newLocalStorage(mynahSettings, pyimplProvider)
}
