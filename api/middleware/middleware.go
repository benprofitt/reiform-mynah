// Copyright (c) 2022 by Reiform. All Rights Reserved.

package middleware

import (
	"bufio"
	"context"
	"errors"
	"github.com/gorilla/mux"
	"log"
	"net"
	"net/http"
)

//response for logging
type logResponse struct {
	http.ResponseWriter
	status int
}

//write response, record status for logging
func (r *logResponse) WriteHeader(stat int) {
	r.status = stat
	r.ResponseWriter.WriteHeader(stat)
}

//implement the hijacker interface for websocket connection upgrade
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

		//handle the request
		handler.ServeHTTP(&res, request)

		//log the result
		log.Printf("%s %s %s %d",
			request.Method,
			request.URL.Path,
			request.Proto,
			res.status)
	})
}

//handle cors
func (r *MynahRouter) corsMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//set the CORS header
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
			if user, getErr := r.dbProvider.GetUserForAuth(&uuid); getErr == nil {
				//call the handler, pass the authenticated user
				handler.ServeHTTP(writer, request.WithContext(
					context.WithValue(request.Context(), contextUserKey, user)))

			} else {
				log.Printf("auth middleware failed to get user from database: %s", getErr)
				writer.WriteHeader(http.StatusUnauthorized)
			}

		} else {
			log.Printf("invalid user authentication: %s", authErr)
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

//load requested project and check for user permissions
//Endpoint path must have {project}
func (r *MynahRouter) projectMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the project key from the request path
		if projectId, ok := mux.Vars(request)[projectKey]; ok {
			//get the user (already authenticated)
			user := GetUserFromRequest(request)
			//request the project from the database
			if project, projectErr := r.dbProvider.GetProject(&projectId, user); projectErr == nil {
				//execute the handler, adding the project as context
				handler.ServeHTTP(writer, request.WithContext(
					context.WithValue(request.Context(), contextProjectKey, &project)))

			} else {
				log.Printf("error retrieving project %s: %s", projectId, projectErr)
				writer.WriteHeader(http.StatusBadRequest)
			}
		} else {
			log.Printf("project request path missing %s", projectKey)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
