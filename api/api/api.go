// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"
)

//Register all api routes
func RegisterRoutes(router *middleware.MynahRouter,
	dbProvider db.DBProvider,
	authProvider auth.AuthProvider,
	storageProvider storage.StorageProvider,
	pythonProvider python.PythonProvider,
	wsProvider websockets.WebSocketProvider,
	asyncProvider async.AsyncProvider,
	settings *settings.MynahSettings) error {

	//register graphql routes
	if gqlErr := registerGQLRoutes(router, dbProvider); gqlErr != nil {
		return gqlErr
	}

	//register the websocket endpoint
	router.HandleHTTPRequest("websocket", wsProvider.ServerHandler())

	//register the file upload endpoint
	router.HandleHTTPRequest("upload", handleFileUpload(settings, dbProvider, storageProvider))

	//register admin endpoints
	router.HandleAdminRequest("POST", "user/create", adminCreateUser(dbProvider, authProvider))

	return nil
}
