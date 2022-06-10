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

	router.HandleHTTPRequest("GET", "file/list", handleListFileMetadata(dbProvider))
	router.HandleHTTPRequest("GET", fmt.Sprintf("file/{%s}/{%s}", fileKey, fileVersionIdKey), handleViewFile(dbProvider, storageProvider))
	router.HandleHTTPRequest("POST", "upload", handleFileUpload(settings, dbProvider, storageProvider, pyImplProvider))

	//register the ic process job endpoint
	router.HandleHTTPRequest("POST", "dataset/ic/process/start", icProcessJob(dbProvider, asyncProvider, pyImplProvider, storageProvider))

	router.HandleHTTPRequest("GET", "dataset/list", allDatasetList(dbProvider))

	router.HandleHTTPRequest("POST", "dataset/ic/create", icDatasetCreate(dbProvider))
	router.HandleHTTPRequest("POST", fmt.Sprintf("dataset/ic/{%s}/export", datasetIdKey), icDatasetExport(dbProvider, storageProvider))
	router.HandleHTTPRequest("GET", "dataset/ic/list", icDatasetList(dbProvider))
	router.HandleHTTPRequest("GET", fmt.Sprintf("dataset/ic/{%s}", datasetIdKey), icDatasetGet(dbProvider))
	router.HandleHTTPRequest("GET", fmt.Sprintf("dataset/ic/{%s}/report", datasetIdKey), icProcessReportView(dbProvider))

	//router.HandleHTTPRequest("POST", "dataset/od/create", ocDatasetCreate(dbProvider))
	router.HandleHTTPRequest("GET", "dataset/od/list", odDatasetList(dbProvider))
	router.HandleHTTPRequest("GET", fmt.Sprintf("dataset/od/{%s}", datasetIdKey), odDatasetGet(dbProvider))

	router.HandleAdminRequest("POST", "user/create", adminCreateUser(dbProvider, authProvider))

	router.HandleHTTPRequest("GET", fmt.Sprintf("task/status/{%s}", taskIdKey), getAsyncTaskStatus(asyncProvider))
	return nil
}
