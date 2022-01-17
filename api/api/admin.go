package api

import (
	"encoding/json"
	"log"
	"net/http"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
)

//Create a new user
func adminCreateUser(dbProvider db.DBProvider, authProvider auth.AuthProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the admin user from context
		admin := middleware.GetUserFromRequest(request)

		var req adminCreateUserRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//create the user
		if user, jwt, err := authProvider.CreateUser(); err != nil {
			//update attributes
			user.NameFirst = req.nameFirst
			user.NameLast = req.nameLast

			//add the user to the database
			if createErr := dbProvider.CreateUser(user, admin); createErr != nil {
				log.Printf("failed to add new user to database %s", err)
				writer.WriteHeader(http.StatusBadRequest)
				return
			}

			//the response to return to the frontend
			res := adminCreateUserResponse{
				jwt:  jwt,
				user: *user,
			}

			//write the response
			if jsonResp, jsonErr := json.Marshal(&res); jsonErr == nil {
				writer.Write(jsonResp)
				//respond with json
				writer.Header().Set("Content-Type", "application/json")

			} else {
				log.Printf("failed to generate json response %s", jsonErr)
				writer.WriteHeader(http.StatusInternalServerError)
			}

		} else {
			log.Printf("failed to create new user %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
