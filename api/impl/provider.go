// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/mynahExec"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
)

// NewImplProvider create a new provider
func NewImplProvider(mynahSettings *settings.MynahSettings,
	dbProvider db.DBProvider,
	storageProvider storage.StorageProvider,
	mynahExecutor mynahExec.MynahExecutor) ImplProvider {
	return &localImplProvider{
		mynahSettings:   mynahSettings,
		dbProvider:      dbProvider,
		storageProvider: storageProvider,
		mynahExecutor:   mynahExecutor,
	}
}
