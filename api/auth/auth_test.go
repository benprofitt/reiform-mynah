package auth

import (
	"net/http"
	"reiform.com/mynah/settings"
	"testing"
)

//TODO run tests for both auth providers by changing the settings and passing in

//Test the behavior of authentication
func TestJWTAuth(t *testing.T) {
	testSettings := settings.DefaultSettings()
	//load the auth provider
	authProvider, apErr := NewAuthProvider(testSettings)
	if apErr != nil {
		t.Errorf("error loading auth provider: %s", apErr)
		return
	}
	defer authProvider.Close()

	//create a new user
	user, jwt, userErr := authProvider.CreateUser()
	if userErr != nil {
		t.Errorf("error creating user: %s", userErr)
		return
	}

	//create a mock request
	req := http.Request{
		Header: make(map[string][]string),
	}
	req.Header.Add(testSettings.AuthSettings.JwtHeader, jwt)

	//authenticate the request
	uuid, authErr := authProvider.IsAuthReq(&req)
	if authErr != nil {
		t.Errorf("error authenticating request: %s", authErr)
		return
	}

	//verify the uuid
	if uuid != user.Uuid {
		t.Errorf("uuids don't match (%s != %s)", uuid, user.Uuid)
	}
}
