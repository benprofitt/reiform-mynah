// Copyright (c) 2022 by Reiform. All Rights Reserved.

package auth

import (
	"net/http"
	"reiform.com/mynah/model"
)

//Defines the interface the auth client must implement
type AuthProvider interface {
	//Generate a jwt for the user
	GetUserAuth(*model.MynahUser) (string, error)
	//Takes an http request and checks whether the request is correctly
	//authenticated
	IsAuthReq(*http.Request) (string, error)
	//close the auth provider
	Close()
}
