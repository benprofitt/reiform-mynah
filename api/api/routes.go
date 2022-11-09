// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"fmt"
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/impl"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"
)

// RegisterRoutes Register all api routes
func RegisterRoutes(router *middleware.MynahRouter,
	dbProvider db.DBProvider,
	authProvider auth.AuthProvider,
	storageProvider storage.StorageProvider,
	implProvider impl.ImplProvider,
	wsProvider websockets.WebSocketProvider,
	asyncProvider async.AsyncProvider,
	settings *settings.MynahSettings) error {

	//register the websocket endpoint
	router.HandleHTTPRequest("GET", "websocket", wsProvider.ServerHandler())

	router.HandleHTTPRequest("GET", "file/list", handleListFileMetadata(dbProvider))
	router.HandleHTTPRequest("GET", fmt.Sprintf("file/{%s}/{%s}", FileKey, FileVersionIdKey), handleViewFile(dbProvider, storageProvider))
	router.HandleHTTPRequest("POST", "upload", handleFileUpload(settings, dbProvider, storageProvider))

	//register the ic process job endpoint
	router.HandleHTTPRequest("POST", "dataset/ic/process/start", icProcessJob(dbProvider, asyncProvider, implProvider))

	router.HandleHTTPRequest("GET", "dataset/list", allDatasetList(dbProvider))

	router.HandleHTTPRequest("POST", "dataset/ic/create", icDatasetCreate(dbProvider, storageProvider, implProvider))
	router.HandleHTTPRequest("POST", fmt.Sprintf("dataset/ic/{%s}/export", DatasetIdKey), icDatasetExport(dbProvider, storageProvider))
	router.HandleHTTPRequest("GET", "dataset/ic/list", icDatasetList(dbProvider))
	router.HandleHTTPRequest("GET", fmt.Sprintf("dataset/ic/{%s}", DatasetIdKey), icDatasetGet(dbProvider))

	router.HandleHTTPRequest("GET", fmt.Sprintf("data/json/{%s}", DatasetIdKey), getDataJSON(dbProvider))

	//router.HandleHTTPRequest("POST", "dataset/od/create", ocDatasetCreate(dbProvider))
	router.HandleHTTPRequest("GET", "dataset/od/list", odDatasetList(dbProvider))
	router.HandleHTTPRequest("GET", fmt.Sprintf("dataset/od/{%s}", DatasetIdKey), odDatasetGet(dbProvider))

	router.HandleAdminRequest("POST", "user/create", adminCreateUser(dbProvider, authProvider))

	router.HandleHTTPRequest("GET", "task/list", listAsyncTasks(asyncProvider))
	router.HandleHTTPRequest("GET", fmt.Sprintf("task/status/{%s}", TaskIdKey), getAsyncTaskStatus(asyncProvider))
	return nil
}
