// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"context"
	"fmt"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"net/http"
	"reiform.com/mynah-api/api/middleware"
	"reiform.com/mynah-api/services/log"
	"reiform.com/mynah-api/settings"
	"time"
)

// MynahRouter serves mynah api routes
type MynahRouter struct {
	e *gin.Engine
	s *http.Server
}

// NewMynahRouter creates a new router
func NewMynahRouter() *MynahRouter {
	gin.SetMode(gin.ReleaseMode)
	e := gin.New()
	e.Use(middleware.Logger)
	e.Use(middleware.Auth)
	e.Use(gin.Recovery())
	e.Use(cors.New(cors.Config{
		AllowOrigins:     settings.GlobalSettings.CORSAllowOrigins,
		AllowMethods:     settings.GlobalSettings.CORSAllowMethods,
		AllowHeaders:     settings.GlobalSettings.CORSAllowHeaders,
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	return &MynahRouter{
		e: e,
	}
}

// ListenAndServe registers and serves routes
func (r *MynahRouter) ListenAndServe() {
	registerRoutes(r.e)

	r.s = &http.Server{
		Handler:           r.e,
		Addr:              fmt.Sprintf(":%d", settings.GlobalSettings.Port),
		WriteTimeout:      15 * time.Second,
		ReadTimeout:       15 * time.Second,
		IdleTimeout:       15 * time.Second,
		ReadHeaderTimeout: 15 * time.Second,
	}
	log.Info("server starting on %s", r.s.Addr)
	log.Info("server exit: %s", r.s.ListenAndServe())
}

// Close shuts down the server
func (r *MynahRouter) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()
	if err := r.s.Shutdown(ctx); err != nil {
		log.Error("server shutdown error: %s", err)
	}
}
