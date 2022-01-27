// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"testing"
)

//create a new user, make an authenticated create user request
func newUserCreateAuth(t *testing.T,
	mynahSettings *settings.MynahSettings,
	authProvider auth.AuthProvider,
	dbProvider db.DBProvider,
	router *middleware.MynahRouter,
	isAdmin bool) error {

	//create a user
	user, userJwt, userErr := authProvider.CreateUser()
	if userErr != nil {
		return fmt.Errorf("failed to create user %s", userErr)
	}
	user.IsAdmin = isAdmin

	//create an admin to insert the admin (must have distinct id)
	creator := model.MynahUser{
		OrgId:   user.OrgId,
		IsAdmin: true,
	}

	//add to the database
	if dbErr := dbProvider.CreateUser(user, &creator); dbErr != nil {
		return fmt.Errorf("failed to create user %s", dbErr)
	}

	var jsonBody = []byte(`{"name_first":"test", "name_last": "test"}`)

	//try to create a user
	req, reqErr := http.NewRequest("POST", filepath.Join(mynahSettings.ApiPrefix, "admin", "user/create"), bytes.NewBuffer(jsonBody))
	if reqErr != nil {
		return fmt.Errorf("failed to create request %s", reqErr)
	}
	//add auth header
	req.Header.Add(mynahSettings.AuthSettings.JwtHeader, userJwt)
	req.Header.Add("Content-Type", "application/json")

	//create a recorder for the response
	rr := httptest.NewRecorder()

	//make the request
	router.ServeHTTP(rr, req)

	//check the result
	if stat := rr.Code; stat != http.StatusOK {
		return fmt.Errorf("create user returned non-200: %v want %v", stat, http.StatusOK)
	}

	//check that the user was inserted into the database
	var res adminCreateUserResponse

	//attempt to parse the request body
	if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
		return fmt.Errorf("failed to decode response %s", err)
	}

	//check for the user in the database (as a known admin)
	dbUser, dbErr := dbProvider.GetUser(&res.User.Uuid, &creator)
	if dbErr != nil {
		return fmt.Errorf("new user not found in database %s", dbErr)
	}

	//verify same
	if dbUser.OrgId != res.User.OrgId {
		return fmt.Errorf("user from db (%v) not assigned same org id (%v)", dbUser.OrgId, res.User.OrgId)
	}

	return nil
}

//Test admin endpoints
func TestAPIAdminEndpoints(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//initialize auth
	authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	if authErr != nil {
		t.Errorf("failed to initialize auth %s", authErr)
		return
	}

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	if dbErr != nil {
		t.Errorf("failed to initialize database connection %s", dbErr)
		return
	}

	//initialize router
	router := middleware.NewRouter(mynahSettings, authProvider, dbProvider)
	//handle user creation endpoint
	router.HandleAdminRequest("POST", "user/create", adminCreateUser(dbProvider, authProvider))

	//make a request as an admin
	if err := newUserCreateAuth(t, mynahSettings, authProvider, dbProvider, router, true); err != nil {
		t.Errorf("failed to create user as admin: %s", err)
		return
	}

	//make a request as a non-admin
	if err := newUserCreateAuth(t, mynahSettings, authProvider, dbProvider, router, false); err == nil {
		t.Error("non-admin was allowed to create a new user")
		return
	}

	//TODO more admin endpoints
}

//test upload and download endpoints
func TestAPIFileEndpoints(t *testing.T) {

	//TODO

	// mynahSettings := settings.DefaultSettings()
	//
	// //initialize auth
	// authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	// if authErr != nil {
	//   t.Errorf("failed to initialize auth %s", authErr)
	//   return
	// }
	//
	// //initialize the database connection
	// dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	// if dbErr != nil {
	//   t.Errorf("failed to initialize database connection %s", dbErr)
	//   return
	// }
	//
	// //initialize router
	// router := middleware.NewRouter(mynahSettings, authProvider, dbProvider)
	// //handle upload endpoint
	// router.HandleHTTPRequest("upload", handleFileUpload(settings, dbProvider, storageProvider))

}
