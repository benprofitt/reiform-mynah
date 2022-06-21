// Copyright (c) 2022 by Reiform. All Rights Reserved.

package middleware

import (
	"bufio"
	"context"
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
func (r *MynahRouter) logMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		res := logResponse{
			ResponseWriter: writer,
			status:         200, //success by default
		}

		start := time.Now()

		//handle the request
		handler.ServeHTTP(&res, request)

		duration := time.Since(start)

		//log the result
		log.Infof("%s %s %s %d %v",
			request.Method,
			request.URL.Path,
			request.Proto,
			res.status,
			duration)
	})
}

//handle cors
func (r *MynahRouter) corsMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//set the CORS headers
		writer.Header().Set("Access-Control-Allow-Headers", r.settings.AuthSettings.JwtHeader)
		writer.Header().Set("Access-Control-Allow-Origin", r.settings.CORSAllowOrigin)
		if request.Method == http.MethodOptions {
			return
		}
		//call the next middleware
		handler.ServeHTTP(writer, request)
	})
}

//authenticate requests
func (r *MynahRouter) authenticationMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//check authentication
		if uuid, authErr := r.authProvider.IsAuthReq(request); authErr == nil {
			//get the user from the database
			if user, getErr := r.dbProvider.GetUserForAuth(uuid); getErr == nil {
				//call the handler, pass the authenticated user
				handler.ServeHTTP(writer, request.WithContext(
					context.WithValue(request.Context(), contextUserKey, user)))

			} else {
				log.Errorf("auth middleware failed to get user from database: %s", getErr)
				writer.WriteHeader(http.StatusUnauthorized)
			}

		} else {
			log.Warnf("invalid user authentication: %s", authErr)
			writer.WriteHeader(http.StatusUnauthorized)
		}
	})
}

//verify that this user is an admin
func (r *MynahRouter) adminMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the authenticated user
		user := GetUserFromRequest(request)
		//check if this user has admin priv.
		if user.IsAdmin {
			//execute the handler
			handler.ServeHTTP(writer, request)
		} else {
			writer.WriteHeader(http.StatusUnauthorized)
		}
	})
}
