package api

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
)

//Register all api routes
func RegisterRoutes(router *middleware.MynahRouter,
	dbProvider db.DBProvider,
	storageProvider storage.StorageProvider,
	settings *settings.MynahSettings) {
	//TODO

	//register graphql routes
	registerGQLRoutes(router, dbProvider)
}
