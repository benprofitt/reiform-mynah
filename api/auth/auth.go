package auth

import (
	"reiform.com/mynah/settings"
)

//Create a new auth provider based on the Mynah settings
func NewAuthProvider(mynahSettings *settings.MynahSettings) (AuthProvider, error) {
	//temp
	return &localAuth{}, nil
}