// Copyright (c) 2022 by Reiform. All Rights Reserved.

package auth

import (
	"net/http"
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"testing"
)

//setup and teardown
func TestMain(m *testing.M) {
	dirPath := "data"

	//create the base directory if it doesn't exist
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		log.Fatalf("failed to create directory: %s", dirPath)
	}

	//run tests
	exitVal := m.Run()

	//remove generated
	if err := os.RemoveAll(dirPath); err != nil {
		log.Errorf("failed to clean up after tests: %s", err)
	}

	os.Exit(exitVal)
}

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

	user := model.MynahUser{
		Uuid: model.NewMynahUuid(),
	}

	//create a new user
	jwt, userErr := authProvider.GetUserAuth(&user)
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
	userUuid, authErr := authProvider.IsAuthReq(&req)
	if authErr != nil {
		t.Errorf("error authenticating request: %s", authErr)
		return
	}

	//verify the uuid
	if userUuid != user.Uuid {
		t.Errorf("uuids don't match (%s != %s)", userUuid, user.Uuid)
	}
}
