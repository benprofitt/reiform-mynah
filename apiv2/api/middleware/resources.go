// Copyright (c) 2023 by Reiform. All Rights Reserved.

package middleware

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"net/http"
	dataset_model "reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/models/types"
	mynah_context "reiform.com/mynah-api/services/context"
	"reiform.com/mynah-api/services/log"
	"reiform.com/mynah-api/settings"
	"strconv"
)

const (
	datasetKey           = "dataset"
	datasetVersionKey    = "dataset_version"
	paginationOptionsKey = "pagination_options"
	appContextKey        = "app_context"
)

// DatasetMiddleware adds a given dataset to the context
func DatasetMiddleware(ctx *gin.Context) {
	datasetId := ctx.Param("dataset_id")
	mynahDataset, found, err := dataset_model.GetMynahDataset(db.NewContext(), types.MynahUuid(datasetId))
	if err != nil {
		log.Info("failed to get dataset (%s): %s", datasetId, err)
		ctx.Status(http.StatusInternalServerError)
		ctx.Abort()
		return
	}
	if !found {
		log.Info("no such dataset: %s", datasetId)
		ctx.Status(http.StatusNotFound)
		ctx.Abort()
		return
	}

	ctx.Set(datasetKey, mynahDataset)
}

// GetDatasetFromContext gets a dataset from gin context
func GetDatasetFromContext(ctx *gin.Context) *dataset_model.MynahDataset {
	dataset, _ := ctx.Get(datasetKey)
	return dataset.(*dataset_model.MynahDataset)
}

// DatasetVersionMiddleware adds a given dataset version to the context
func DatasetVersionMiddleware(ctx *gin.Context) {
	dataset := GetDatasetFromContext(ctx)
	versionId := ctx.Param("version_id")
	// TODO actually look at the type in the dataset
	datasetVersion, found, err := dataset_model.GetMynahICDatasetVersion(db.NewContext(), dataset.DatasetId, types.MynahUuid(versionId))
	if err != nil {
		log.Info("failed to get dataset version (%s): %s", versionId, err)
		ctx.Status(http.StatusInternalServerError)
		ctx.Abort()
		return
	}

	if !found {
		log.Info("no such dataset (%s) version: %s", dataset.DatasetId, versionId)
		ctx.Status(http.StatusNotFound)
		ctx.Abort()
		return
	}

	ctx.Set(datasetVersionKey, datasetVersion)
}

// GetICDatasetVersionFromContext gets a dataset version from gin context
func GetICDatasetVersionFromContext(ctx *gin.Context) *dataset_model.MynahICDatasetVersion {
	dataset, _ := ctx.Get(datasetVersionKey)
	return dataset.(*dataset_model.MynahICDatasetVersion)
}

// GetAppContext gets app context from the gin context
func GetAppContext(ctx *gin.Context) *mynah_context.Context {
	appCtx, _ := ctx.Get(appContextKey)
	return appCtx.(*mynah_context.Context)
}

// PaginationMiddleware parses pagination info
func PaginationMiddleware(ctx *gin.Context) {
	pageS := ctx.DefaultQuery("page", "0")
	pageSizeS := ctx.DefaultQuery("page_size", fmt.Sprintf("%d", settings.GlobalSettings.DefaultPageSize))
	page, err := strconv.Atoi(pageS)
	if err != nil {
		log.Error("failed to parse 'page' as int: %s", err)
		ctx.Status(http.StatusBadRequest)
		ctx.Abort()
		return
	}
	pageSize, err := strconv.Atoi(pageSizeS)
	if err != nil {
		log.Error("failed to parse 'page_size' as int: %s", err)
		ctx.Status(http.StatusBadRequest)
		ctx.Abort()
		return
	}

	ctx.Set(paginationOptionsKey, &db.PaginationOptions{
		Page:     page,
		PageSize: pageSize,
	})
}

// GetPaginationOptionsFromContext gets pagination options from context
func GetPaginationOptionsFromContext(ctx *gin.Context) *db.PaginationOptions {
	opts, _ := ctx.Get(paginationOptionsKey)
	return opts.(*db.PaginationOptions)
}
