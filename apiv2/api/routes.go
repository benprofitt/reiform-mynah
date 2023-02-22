// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
	"path"
	"reiform.com/mynah-api/settings"
)

func registerRoutes(e *gin.Engine) {
	// API Routes
	apiRoutes := e.Group("api")
	v2Routes := apiRoutes.Group("v2")
	v2Routes.Group("user").
		POST("create", UserCreate)
	v2Routes.Group("dataset").
		POST("create", DatasetCreate)
	v2Routes.Group("raw")
	v2Routes.Group("file")

	// Static Routes
	e.GET("/", func(c *gin.Context) {
		c.Redirect(http.StatusMovedPermanently, path.Join(settings.GlobalSettings.StaticPrefix, "index.html"))
	})
	e.Static(settings.GlobalSettings.StaticPrefix, settings.GlobalSettings.StaticResourcesPath)
}
