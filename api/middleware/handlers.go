// Copyright (c) 2022 by Reiform. All Rights Reserved.

package middleware

import (
	"context"
	"fmt"
	"github.com/gorilla/mux"
	"net/http"
	"path"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"time"
)

type HandlerFunc func(*Context)

func (f HandlerFunc) ServeHTTP(ctx *Context) {
	f(ctx)
}

// NewRouter Create a new router
func NewRouter(mynahSettings *settings.MynahSettings,
	authProvider auth.AuthProvider,
	dbProvider db.DBProvider,
	storageProvider storage.StorageProvider) *MynahRouter {
	return &MynahRouter{
		mux.NewRouter(),
		nil,
		mynahSettings,
		authProvider,
		dbProvider,
		storageProvider,
	}
}

// HttpMiddleware wraps the given function in basic request middleware
func (r *MynahRouter) HttpMiddleware(handler HandlerFunc) http.HandlerFunc {
	return r.ctxMiddleware(r.logMiddleware(r.corsMiddleware(r.authenticationMiddleware(handler))))
}

// HandleHTTPRequest handle a basic http request (authenticated user passed in request context)
func (r *MynahRouter) HandleHTTPRequest(method, urlPath string, handler HandlerFunc) *mux.Route {
	return r.HandleFunc(path.Join(r.settings.ApiPrefix, urlPath),
		r.HttpMiddleware(handler)).Methods(method, http.MethodOptions)
}

// HandleAdminRequest Handle an admin request (passes authenticated admin)
func (r *MynahRouter) HandleAdminRequest(method string, urlPath string, handler HandlerFunc) *mux.Route {
	return r.HandleFunc(path.Join(r.settings.ApiPrefix, "admin", urlPath),
		r.HttpMiddleware(r.adminMiddleware(handler))).Methods(method, http.MethodOptions)
}

// ListenAndServe start server
func (r *MynahRouter) ListenAndServe() {
	//serve static resources
	r.serveStaticSite()

	//set the 404 handler
	r.NotFoundHandler = r.ctxMiddleware(r.logMiddleware(func(ctx *Context) {
		ctx.Writer.WriteHeader(http.StatusNotFound)
	}))

	r.server = &http.Server{
		Handler:           r,
		Addr:              fmt.Sprintf(":%d", r.settings.Port),
		WriteTimeout:      15 * time.Second,
		ReadTimeout:       15 * time.Second,
		IdleTimeout:       60 * time.Second,
		ReadHeaderTimeout: 15 * time.Second,
	}
	log.Infof("server starting on %s", r.server.Addr)
	log.Warnf("server exit: %s", r.server.ListenAndServe())
}

// Close Shutdown the server
func (r *MynahRouter) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()
	if err := r.server.Shutdown(ctx); err != nil {
		log.Errorf("server shutdown error: %s", err)
	}
}
