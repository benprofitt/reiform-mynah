// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
	"reiform.com/mynah-api/api/middleware"
	"reiform.com/mynah-api/settings"
)

func registerRoutes(e *gin.Engine) {
	// API Routes
	apiRoutes := e.Group("api")
	v2Routes := apiRoutes.Group("v2")

	v2Routes.Group("user").
		POST("create", UserCreate)

	datasetGroup := v2Routes.Group("dataset")
	{
		datasetGroup.POST("create", DatasetCreate)
		datasetGroup.GET("list", middleware.PaginationMiddleware, DatasetList)

		specificDatasetGroup := datasetGroup.Group(":dataset_id", middleware.DatasetMiddleware)
		{
			specificDatasetGroup.GET("", DatasetGet)
			specificDatasetGroup.GET("version/refs", DatasetVersionRefs)
			specificDatasetGroup.GET("version/list", middleware.PaginationMiddleware, DatasetVersionList)

			versionDatasetGroup := specificDatasetGroup.Group("version/:version_id", middleware.DatasetVersionMiddleware)
			versionDatasetGroup.GET("", DatasetVersionGet)
			versionDatasetGroup.POST("upload", DatasetUploadFile)
		}
	}

	v2Routes.Group("raw")
	v2Routes.Group("file")

	// Static Routes
	e.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, settings.GlobalSettings.StaticPrefix)
	})
	e.Static(settings.GlobalSettings.StaticPrefix, settings.GlobalSettings.StaticResourcesPath)
}
