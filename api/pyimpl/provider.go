// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
)

// NewPyImplProvider create a new provider
func NewPyImplProvider(mynahSettings *settings.MynahSettings,
	pythonProvider python.PythonProvider,
	dbProvider db.DBProvider,
	storageProvider storage.StorageProvider) PyImplProvider {

	log.Infof("using python impl module %s", mynahSettings.PythonSettings.ModuleName)

	return &localImplProvider{
		pythonProvider:  pythonProvider,
		dbProvider:      dbProvider,
		storageProvider: storageProvider,
		mynahSettings:   mynahSettings,
	}
}
