// Copyright (c) 2022 by Reiform. All Rights Reserved.

package middleware

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/gorilla/mux"
	"net/http"
	"os"
	"path/filepath"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"time"
)

type ctxKey string

const contextUserKey ctxKey = "user"
const contextProjectKey ctxKey = "project"
const projectKey string = "project"
const fileKey string = "file"

// MynahUserHandler Handler for a request that requires the user
type MynahUserHandler func(user *model.MynahUser) (*Response, error)

// MynahProjectHandler Handler for a request that requires the user and a project
type MynahProjectHandler func(user *model.MynahUser, project *model.MynahProject) (*Response, error)

// GetUserFromRequest extract the user from context (can be used externally for basic http requests)
func GetUserFromRequest(request *http.Request) *model.MynahUser {
	return request.Context().Value(contextUserKey).(*model.MynahUser)
}

//extract the project from the request
func getProjectFromRequest(request *http.Request) *model.MynahProject {
	return request.Context().Value(contextProjectKey).(*model.MynahProject)
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

//handler that passes the user to the mynah handler as well as the project
func (r *MynahRouter) projectHandler(handler MynahProjectHandler) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the user and project from context
		user := GetUserFromRequest(request)
		project := getProjectFromRequest(request)

		//call the handler
		if res, handlerErr := handler(user, project); handlerErr == nil {
			//write the status
			writer.WriteHeader(res.Status)

			if res.Body != nil {
				//serialize as json
				if jsonResp, jsonErr := json.Marshal(res.Body); jsonErr == nil {
					if _, err := writer.Write(jsonResp); err == nil {
						//respond with json
						writer.Header().Set("Content-Type", "application/json")
					} else {
						log.Errorf("failed to write json response for request: %s", err)
						writer.WriteHeader(http.StatusInternalServerError)
					}

				} else {
					log.Errorf("failed to generate json response %s", jsonErr)
					writer.WriteHeader(http.StatusInternalServerError)
				}
			}
		} else {
			log.Errorf("handler returned error %s", handlerErr)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

//handler that loads a file and serves the contents
func (r *MynahRouter) fileHandler(writer http.ResponseWriter, request *http.Request) {
	//get the user from context
	user := GetUserFromRequest(request)

	//get the file id
	if fileId, ok := mux.Vars(request)[fileKey]; ok {
		//load the file metadata
		if file, fileErr := r.dbProvider.GetFile(&fileId, user); fileErr == nil {
			//serve the file contents
			storeErr := r.storageProvider.GetStoredFile(file, func(path *string) error {
				//open the file
				osFile, osErr := os.Open(*path)
				if osErr != nil {
					return fmt.Errorf("failed to open file %s: %s", file.Uuid, osErr)
				}

				defer func() {
					if err := osFile.Close(); err != nil {
						log.Errorf("error closing file %s: %s", file.Uuid, err)
					}
				}()

				modTime := time.Unix(file.Created, 0)

				//determine the last modified time
				http.ServeContent(writer, request, file.Name, modTime, osFile)
				return nil
			})

			if storeErr != nil {
				log.Errorf("error writing file to response: %s", storeErr)
				writer.WriteHeader(http.StatusInternalServerError)
			}

		} else {
			log.Warnf("error retrieving file %s: %s", fileId, fileErr)
			writer.WriteHeader(http.StatusBadRequest)
		}

	} else {
		log.Errorf("file request path missing %s", fileKey)
		writer.WriteHeader(http.StatusInternalServerError)
	}
}

// HttpMiddleware wraps the given function in basic request middleware
func (r *MynahRouter) HttpMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return r.logMiddleware(r.corsMiddleware(r.authenticationMiddleware(handler)))
}

// HandleHTTPRequest handle a basic http request (authenticated user passed in request context)
func (r *MynahRouter) HandleHTTPRequest(method, path string, handler http.HandlerFunc) {
	r.HandleFunc(filepath.Join(r.settings.ApiPrefix, path),
		r.HttpMiddleware(handler)).Methods(method, http.MethodOptions)
}

// HandleAdminRequest Handle an admin request (passes authenticated admin)
func (r *MynahRouter) HandleAdminRequest(method string, path string, handler http.HandlerFunc) {
	r.HandleFunc(filepath.Join(r.settings.ApiPrefix, "admin", path),
		r.HttpMiddleware(r.adminMiddleware(handler))).Methods(method, http.MethodOptions)
}

// HandleProjectRequest handle a project request (loads project, passes to handler)
func (r *MynahRouter) HandleProjectRequest(method string, path string, handler MynahProjectHandler) {
	r.HandleFunc(filepath.Join(r.settings.ApiPrefix, fmt.Sprintf("project/{%s}", projectKey), path),
		r.HttpMiddleware(r.projectMiddleware(r.projectHandler(handler)))).Methods(method, http.MethodOptions)
}

// HandleFileRequest handle a request for a file
func (r *MynahRouter) HandleFileRequest(path string) {
	r.HandleFunc(filepath.Join(r.settings.ApiPrefix, path, fmt.Sprintf("{%s}", fileKey)),
		r.HttpMiddleware(r.fileHandler)).Methods("GET", http.MethodOptions)
}

// ListenAndServe start server
func (r *MynahRouter) ListenAndServe() {
	//serve static resources
	r.serveStaticSite()

	r.server = &http.Server{
		Handler:      r,
		Addr:         fmt.Sprintf(":%d", r.settings.Port),
		WriteTimeout: 15 * time.Second,
		ReadTimeout:  15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	log.Infof("server starting on %s", r.server.Addr)
	log.Warnf("server exit: %s", r.server.ListenAndServe())
}

//Shutdown the server
func (r *MynahRouter) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()
	if err := r.server.Shutdown(ctx); err != nil {
		log.Errorf("server shutdown error: %s", err)
	}
}
