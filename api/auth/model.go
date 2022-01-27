// Copyright (c) 2022 by Reiform. All Rights Reserved.

package auth

import (
	"net/http"
	"reiform.com/mynah/model"
)

//Defines the interface the auth client must implement
type AuthProvider interface {
	//Create a new user, returns the user and initial jwt
	CreateUser() (*model.MynahUser, string, error)
	//Takes an http request and checks whether the request is correctly
	//authenticated
	IsAuthReq(*http.Request) (string, error)
	//close the auth provider
	Close()
}
