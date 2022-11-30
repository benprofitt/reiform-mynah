// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"net/http"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

// adminCreateUser creates a new user
func adminCreateUser(dbProvider db.DBProvider, authProvider auth.AuthProvider) middleware.HandlerFunc {
	return func(ctx *middleware.Context) {
		var req AdminCreateUserRequest

		//attempt to parse the request body
		if err := ctx.ReadJson(&req); err != nil {
			ctx.Error(http.StatusBadRequest, "failed to parse json: %s", err)
			return
		}

		//try to create the user
		user, err := dbProvider.CreateUser(ctx.User, func(user *model.MynahUser) error {
			//update attributes
			user.NameFirst = req.NameFirst
			user.NameLast = req.NameLast
			return nil
		})

		if err != nil {
			ctx.Error(http.StatusBadRequest, "failed to add new user to database %s", err)
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
			if err := ctx.WriteJson(&res); err != nil {
				ctx.Error(http.StatusInternalServerError, "failed to write response as json: %s", err)
				return
			}

		} else {
			ctx.Error(http.StatusInternalServerError, "failed to create jwt for user %s: %s", err)
			return
		}
	}
}
