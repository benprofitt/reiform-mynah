package middleware

import (
	"encoding/json"
	"fmt"
	"github.com/gorilla/mux"
	"log"
	"net/http"
	"path/filepath"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"time"
	"context"
)

type ctxKey string

const contextUserKey ctxKey = "user"
const contextProjectKey ctxKey = "project"

//Handler for a request that requires the user
type MynahUserHandler func(user *model.MynahUser) (*Response, error)

//Handler for a request that requires the user and a project
type MynahProjectHandler func(user *model.MynahUser, project *model.MynahProject) (*Response, error)

//extract the user from context (can be used externally for basic http requests)
func GetUserFromRequest(request *http.Request) *model.MynahUser {
	return request.Context().Value(contextUserKey).(*model.MynahUser)
}

//extract the project from the request
func getProjectFromRequest(request *http.Request) *model.MynahProject {
	return request.Context().Value(contextProjectKey).(*model.MynahProject)
}

//Create a new router
func NewRouter(mynahSettings *settings.MynahSettings, authProvider auth.AuthProvider, dbProvider db.DBProvider) *MynahRouter {
	return &MynahRouter{
		mux.NewRouter(),
		nil,
		mynahSettings,
		authProvider,
		dbProvider,
	}
}

//handler that passes user to the mynah handler
func (r *MynahRouter) userHandler(handler MynahUserHandler) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {

		//call the handler
		if res, err := handler(GetUserFromRequest(request)); err == nil {
			//write the status
			writer.WriteHeader(res.Status)

			if res.Body != nil {
				//serialize as json
				if jsonResp, jsonErr := json.Marshal(res.Body); jsonErr == nil {
					writer.Write(jsonResp)
					//respond with json
					writer.Header().Set("Content-Type", "application/json")
				} else {
					log.Printf("failed to generate json response %s", jsonErr)
					writer.WriteHeader(http.StatusInternalServerError)
				}
			}
		} else {
			log.Printf("handler returned error %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
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
					writer.Write(jsonResp)
					//respond with json
					writer.Header().Set("Content-Type", "application/json")

				} else {
					log.Printf("failed to generate json response %s", jsonErr)
					writer.WriteHeader(http.StatusInternalServerError)
				}
			}
		} else {
			log.Printf("handler returned error %s", handlerErr)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

//Handle a request
func (r *MynahRouter) HandleRequest(method string, path string, handler MynahUserHandler) {
	r.HandleFunc(filepath.Join(r.settings.ApiPrefix, path),
		r.logMiddleware(
			r.corsMiddleware(
				r.authenticationMiddleware(r.userHandler(handler))))).Methods(method, http.MethodOptions)
}

//handle a basic http request (authenticated)
func (r *MynahRouter) HandleHTTPRequest(method string, path string, handler http.HandlerFunc) {
	r.HandleFunc(filepath.Join(r.settings.ApiPrefix, path),
		r.logMiddleware(
			r.corsMiddleware(
				r.authenticationMiddleware(handler)))).Methods(method, http.MethodOptions)
}

//Handle an admin request
func (r *MynahRouter) HandleAdminRequest(method string, path string, handler MynahUserHandler) {
	r.HandleFunc(filepath.Join(r.settings.ApiPrefix, path),
		r.logMiddleware(
			r.corsMiddleware(
				r.authenticationMiddleware(
					r.adminMiddleware(r.userHandler(handler)))))).Methods(method, http.MethodOptions)
}

//handle a project request (loads project, updates for POST requests)
func (r *MynahRouter) HandleProjectRequest(method string, path string, handler MynahProjectHandler) {
	r.HandleFunc(filepath.Join(r.settings.ApiPrefix, fmt.Sprintf("{%s}", projectKey), path),
		r.logMiddleware(
			r.corsMiddleware(
				r.authenticationMiddleware(
					r.projectMiddleware(r.projectHandler(handler)))))).Methods(method, http.MethodOptions)
}

//start server
func (r *MynahRouter) ListenAndServe() {
	r.server = &http.Server{
		Handler:      r,
		Addr:         fmt.Sprintf(":%d", r.settings.Port),
		WriteTimeout: 15 * time.Second,
		ReadTimeout:  15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	log.Fatal(r.server.ListenAndServe())
}

//Shutdown the server
func (r *MynahRouter) Shutdown(ctx context.Context) {
	r.server.Shutdown(ctx)
}