package api

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
)

//Register all api routes
func RegisterRoutes(router *middleware.MynahRouter,
	dbProvider db.DBProvider,
	storageProvider storage.StorageProvider,
	pythonProvider python.PythonProvider,
	settings *settings.MynahSettings) error {
	//TODO

	//register graphql routes
	if gqlErr := registerGQLRoutes(router, dbProvider); gqlErr != nil {
		return gqlErr
	}

	return nil
}
