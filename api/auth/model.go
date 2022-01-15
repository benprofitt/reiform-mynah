package auth

import (
	"reiform.com/mynah/model"
)

//Defines the interface the auth client must implement
type AuthProvider interface {
	//Create a new user, returns the user and initial jwt
	CreateUser() (*model.MynahUser, string, error)
	//Takes a JWT token attached to a request and verifies that the token is
	//valid. If valid, returns the user's uuid. If invalid, returns an error
	IsValidToken(*string) (string, error)
	//close the auth provider
	Close()
}

//local auth client adheres to AuthProvider
type localAuth struct {
	//the jwt key loaded from file
	jwtKey string
}

//external auth client adheres to AuthProvider
type externalAuth struct {
}
