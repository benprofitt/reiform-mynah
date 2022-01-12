package middleware

import (
	"context"
	"github.com/gorilla/mux"
	"log"
	"net/http"
	"reiform.com/mynah/model"
)

const projectKey string = "project"

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

//Log all requests
func (r *MynahRouter) logMiddleware(handler http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		res := logResponse{
			ResponseWriter: writer,
			status:         500,
		}

		//handle the request
		handler.ServeHTTP(&res, request)

		//log the result
		log.Printf("%s %s %s %d (%s)",
			request.Method,
			request.URL.Path,
			request.Proto,
			res.status,
			GetUserFromRequest(request).Uuid)
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
		//TODO check auth token, get user from db
		u := model.MynahUser{}

		//call the handler, pass the authenticated user
		handler.ServeHTTP(writer, request.WithContext(
			context.WithValue(request.Context(), contextUserKey, &u)))
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