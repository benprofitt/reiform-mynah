package api

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/settings"
)

//Register all api routes
func RegisterRoutes(router *middleware.MynahRouter,
										dbProvider db.DBProvider,
										storageProvider storage.StorageProvider,
										settings *settings.MynahSettings) {
	//TODO
}
