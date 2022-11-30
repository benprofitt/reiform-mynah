// Copyright (c) 2022 by Reiform. All Rights Reserved.

package middleware

import (
	"bufio"
	"errors"
	"net"
	"net/http"
	"reiform.com/mynah/log"
	"time"
)

//response for logging
type logResponse struct {
	http.ResponseWriter
	status int
}

// WriteHeader write response, record status for logging
func (r *logResponse) WriteHeader(stat int) {
	r.status = stat
	r.ResponseWriter.WriteHeader(stat)
}

// Hijack implement the hijacker interface for websocket connection upgrade
func (r *logResponse) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	h, ok := r.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, errors.New("hijack not supported")
	}
	return h.Hijack()
}

//Log all requests
func (r *MynahRouter) logMiddleware(handler HandlerFunc) HandlerFunc {
	return func(ctx *Context) {
		res := logResponse{
			ResponseWriter: ctx.Writer,
			status:         200, //success by default
		}

		ctx.Writer = &res

		start := time.Now()

		//handle the request
		handler.ServeHTTP(ctx)

		duration := time.Since(start)

		//log the result
		log.Infof("%s %s %s %d %v",
			ctx.Request.Method,
			ctx.Request.URL.Path,
			ctx.Request.Proto,
			res.status,
			duration)
	}
}

//handle cors
func (r *MynahRouter) corsMiddleware(handler HandlerFunc) HandlerFunc {
	return func(ctx *Context) {
		//set the CORS headers
		// writer.Header().Set("Access-Control-Allow-Headers", r.settings.AuthSettings.JwtHeader)
		ctx.Writer.Header().Set("Access-Control-Allow-Headers", r.settings.CORSAllowHeaders)
		ctx.Writer.Header().Set("Access-Control-Allow-Origin", r.settings.CORSAllowOrigin)
		if ctx.Request.Method == http.MethodOptions {
			return
		}
		//call the next middleware
		handler.ServeHTTP(ctx)
	}
}

//authenticate requests
func (r *MynahRouter) authenticationMiddleware(handler HandlerFunc) HandlerFunc {
	return func(ctx *Context) {
		//check authentication
		if uuid, err := r.authProvider.IsAuthReq(ctx.Request); err == nil {
			//get the user from the database
			if user, err := r.dbProvider.GetUserForAuth(uuid); err == nil {
				ctx.User = user
				handler.ServeHTTP(ctx)
			} else {
				ctx.Error(http.StatusUnauthorized, "auth middleware failed to get user from database: %s", err)
			}

		} else {
			ctx.Error(http.StatusUnauthorized, "invalid user authentication: %s", err)
		}
	}
}

//verify that this user is an admin
func (r *MynahRouter) adminMiddleware(handler HandlerFunc) HandlerFunc {
	return func(ctx *Context) {
		//check if this user has admin priv.
		if ctx.User != nil && ctx.User.IsAdmin {
			//execute the handler
			handler.ServeHTTP(ctx)
		} else {
			ctx.Writer.WriteHeader(http.StatusUnauthorized)
		}
	}
}

// ctxMiddleware upgrades the http handler to a Mynah context handler
func (r *MynahRouter) ctxMiddleware(handler HandlerFunc) http.HandlerFunc {
	return func(writer http.ResponseWriter, request *http.Request) {
		handler(NewContext(writer, request))
	}
}
