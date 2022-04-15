// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"fmt"
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/extensions"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/pyimpl"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"
)

// RegisterRoutes Register all api routes
func RegisterRoutes(router *middleware.MynahRouter,
	dbProvider db.DBProvider,
	authProvider auth.AuthProvider,
	storageProvider storage.StorageProvider,
	pyImplProvider pyimpl.PyImplProvider,
	wsProvider websockets.WebSocketProvider,
	asyncProvider async.AsyncProvider,
	extensionManager *extensions.ExtensionManager,
	settings *settings.MynahSettings) error {

	//register the websocket endpoint
	router.HandleHTTPRequest("GET", "websocket", wsProvider.ServerHandler())

	router.HandleHTTPRequest("GET", fmt.Sprintf("file/{%s}/{%s}", fileKey, fileVersionIdKey), handleViewFile(dbProvider, storageProvider))
	router.HandleHTTPRequest("POST", "upload", handleFileUpload(settings, dbProvider, storageProvider))

	router.HandleHTTPRequest("POST", "icdataset/create", icDatasetCreate(dbProvider, storageProvider))
	router.HandleHTTPRequest("GET", "icdataset/list", icDatasetList(dbProvider))

	//router.HandleHTTPRequest("POST", "ocdataset/create", ocDatasetCreate(dbProvider))
	router.HandleHTTPRequest("GET", "oddataset/list", odDatasetList(dbProvider))

	router.HandleHTTPRequest("GET",
		fmt.Sprintf("icdataset/report/{%s}", icReportKey),
		icDiagnosisReportView(dbProvider))

	//register the ic diagnosis job endpoint
	router.HandleHTTPRequest("POST", "ic/diagnosis/start",
		startICDiagnosisJob(dbProvider, asyncProvider, pyImplProvider, storageProvider))

	//register admin endpoints
	router.HandleAdminRequest("POST", "user/create", adminCreateUser(dbProvider, authProvider))

	return nil
}
