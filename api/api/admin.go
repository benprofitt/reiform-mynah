// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"net/http"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

// adminCreateUser creates a new user
func adminCreateUser(dbProvider db.DBProvider, authProvider auth.AuthProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the admin user from context
		admin := middleware.GetUserFromRequest(request)

		var req AdminCreateUserRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//try to create the user
		user, err := dbProvider.CreateUser(admin, func(user *model.MynahUser) error {
			//update attributes
			user.NameFirst = req.NameFirst
			user.NameLast = req.NameLast
			return nil
		})

		if err != nil {
			log.Errorf("failed to add new user to database %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//get the jwt for the user
		if jwt, err := authProvider.GetUserAuth(user); err == nil {
			//the response to return to the frontend
			res := AdminCreateUserResponse{
				Jwt:  jwt,
				User: *user,
			}

			//write the response
			if err := responseWriteJson(writer, &res); err != nil {
				log.Warnf("failed to write response as json: %s", err)
				writer.WriteHeader(http.StatusInternalServerError)
			}

		} else {
			log.Errorf("failed to create jwt for user %s: %s", user.Uuid, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
	})
}
