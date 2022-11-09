// Copyright (c) 2022 by Reiform. All Rights Reserved.

package auth

import (
	"net/http"
	"reiform.com/mynah/model"
)

// AuthProvider Defines the interface the auth client must implement
type AuthProvider interface {
	// GetUserAuth Generate a jwt for the user
	GetUserAuth(*model.MynahUser) (string, error)
	// IsAuthReq Takes an http request and checks whether the request is correctly
	//authenticated
	IsAuthReq(*http.Request) (model.MynahUuid, error)
	// Close the auth provider
	Close()
}
