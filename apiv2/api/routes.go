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
	//r.e.Group(settings.GlobalSettings.StaticPrefix).
	//	GET(settings.GlobalSettings.StaticPrefix, func(c *gin.Context) {
	//		c.File(path.Join(settings.GlobalSettings.StaticResourcesPath, "index.html"))
	//	}).Static(path.Join(settings.GlobalSettings.BuildAssetsFolder, "*any"), settings.GlobalSettings.StaticResourcesPath)

}
