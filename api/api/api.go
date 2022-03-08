// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/pyimpl"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"
)

//Register all api routes
func RegisterRoutes(router *middleware.MynahRouter,
	dbProvider db.DBProvider,
	authProvider auth.AuthProvider,
	storageProvider storage.StorageProvider,
	pyImplProvider pyimpl.PyImplProvider,
	wsProvider websockets.WebSocketProvider,
	asyncProvider async.AsyncProvider,
	settings *settings.MynahSettings) error {

	//register graphql routes
	if gqlErr := registerGQLRoutes(router, dbProvider); gqlErr != nil {
		return gqlErr
	}

	//register the websocket endpoint
	router.HandleHTTPRequest("GET", "websocket", wsProvider.ServerHandler())

	//register the file viewer endpoint
	router.HandleFileRequest("file")

	router.HandleHTTPRequest("POST", "upload", handleFileUpload(settings, dbProvider, storageProvider))
	router.HandleHTTPRequest("POST", "icdataset/create", icDatasetCreate(dbProvider))
	router.HandleHTTPRequest("POST", "icproject/create", icProjectCreate(dbProvider))

	//register the ic diagnosis job endpoint
	router.HandleHTTPRequest("POST", "ic/diagnosis/start",
		startICDiagnosisJob(dbProvider, asyncProvider, pyImplProvider, storageProvider))

	//register admin endpoints
	router.HandleAdminRequest("POST", "user/create", adminCreateUser(dbProvider, authProvider))

	return nil
}
