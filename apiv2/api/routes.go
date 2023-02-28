// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
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
		datasetGroup.GET("list", DatasetList)

		specificDatasetGroup := datasetGroup.Group(":dataset_id")
		{
			specificDatasetGroup.GET("", DatasetGet)

			versionDatasetGroup := specificDatasetGroup.Group("version/:version_id")
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
