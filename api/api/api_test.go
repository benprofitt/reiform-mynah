// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"testing"
)

//create a new user, make an authenticated create user request
func newUserCreateAuth(t *testing.T,
	mynahSettings *settings.MynahSettings,
	authProvider auth.AuthProvider,
	dbProvider db.DBProvider,
	router *middleware.MynahRouter,
	isAdmin bool) (*model.MynahUser, string, error) {

	//create a user
	user, userJwt, userErr := authProvider.CreateUser()
	if userErr != nil {
		return nil, "", fmt.Errorf("failed to create user %s", userErr)
	}
	user.IsAdmin = isAdmin

	//create an admin to insert the admin (must have distinct id)
	creator := model.MynahUser{
		OrgId:   user.OrgId,
		IsAdmin: true,
	}

	//add to the database
	if dbErr := dbProvider.CreateUser(user, &creator); dbErr != nil {
		return nil, "", fmt.Errorf("failed to create user %s", dbErr)
	}

	var jsonBody = []byte(`{"name_first":"test", "name_last": "test"}`)

	//try to create a user
	req, reqErr := http.NewRequest("POST", filepath.Join(mynahSettings.ApiPrefix, "admin", "user/create"), bytes.NewBuffer(jsonBody))
	if reqErr != nil {
		return nil, "", fmt.Errorf("failed to create request %s", reqErr)
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
		return nil, "", fmt.Errorf("create user returned non-200: %v want %v", stat, http.StatusOK)
	}

	//check that the user was inserted into the database
	var res adminCreateUserResponse

	//attempt to parse the request body
	if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
		return nil, "", fmt.Errorf("failed to decode response %s", err)
	}

	//check for the user in the database (as a known admin)
	dbUser, dbErr := dbProvider.GetUser(&res.User.Uuid, &creator)
	if dbErr != nil {
		return nil, "", fmt.Errorf("new user not found in database %s", dbErr)
	}

	//verify same
	if dbUser.OrgId != res.User.OrgId {
		return nil, "", fmt.Errorf("user from db (%v) not assigned same org id (%v)", dbUser.OrgId, res.User.OrgId)
	}

	return user, userJwt, nil
}

//Test admin endpoints
func TestAPIAdminEndpoints(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//initialize auth
	authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	if authErr != nil {
		t.Fatalf("failed to initialize auth %s", authErr)
	}
	defer authProvider.Close()

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	if dbErr != nil {
		t.Fatalf("failed to initialize database connection %s", dbErr)
	}
	defer dbProvider.Close()

	//initialize storage
	storageProvider, storageErr := storage.NewStorageProvider(mynahSettings)
	if storageErr != nil {
		t.Fatalf("failed to initialize storage %s", storageErr)
	}
	defer storageProvider.Close()

	//initialize router
	router := middleware.NewRouter(mynahSettings, authProvider, dbProvider, storageProvider)

	//handle user creation endpoint
	router.HandleAdminRequest("POST", "user/create", adminCreateUser(dbProvider, authProvider))

	//make a request as an admin
	if _, _, err := newUserCreateAuth(t, mynahSettings, authProvider, dbProvider, router, true); err != nil {
		t.Fatalf("failed to create user as admin: %s", err)
	}

	//make a request as a non-admin
	if _, _, err := newUserCreateAuth(t, mynahSettings, authProvider, dbProvider, router, false); err == nil {
		t.Fatalf("non-admin was allowed to create a new user")
	}

	//TODO more admin endpoints
}

func TestFileGetEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//initialize auth
	authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	if authErr != nil {
		t.Fatalf("failed to initialize auth %s", authErr)
	}
	defer authProvider.Close()

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	if dbErr != nil {
		t.Fatalf("failed to initialize database connection %s", dbErr)
	}
	defer dbProvider.Close()

	//initialize storage
	storageProvider, storageErr := storage.NewStorageProvider(mynahSettings)
	if storageErr != nil {
		t.Fatalf("failed to initialize storage %s", storageErr)
	}
	defer storageProvider.Close()

	//initialize router
	router := middleware.NewRouter(mynahSettings, authProvider, dbProvider, storageProvider)
	//handle user creation
	router.HandleAdminRequest("POST", "user/create", adminCreateUser(dbProvider, authProvider))
	//handle user creation endpoint
	router.HandleFileRequest("file")

	//create a user
	user, jwt, err := newUserCreateAuth(t, mynahSettings, authProvider, dbProvider, router, true)
	if err != nil {
		t.Fatalf("failed to create user as admin: %s", err)
	}

	var file model.MynahFile
	file.Name = "test.txt"

	//create a file
	if createErr := dbProvider.CreateFile(&file, user); createErr != nil {
		t.Fatalf("failed to create file in database: %s", createErr)
	}

	fileContents := "test contents"
	expectedType := "text/plain; charset=utf-8"

	//create the file in storage
	storeErr := storageProvider.StoreFile(&file, func(f *os.File) error {
		//write contents to the file
		_, err := f.WriteString(fileContents)
		return err
	})

	if storeErr != nil {
		t.Fatalf("failed to write to file")
	}

	//update the file in the database
	if updateErr := dbProvider.UpdateFile(&file, user, "path"); updateErr != nil {
		t.Fatalf("failed to update file path in database: %s", updateErr)
	}

	//request the file
	req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, "file", file.Uuid), nil)
	if reqErr != nil {
		t.Fatalf("failed to create request %s", reqErr)
	}
	//add auth header
	req.Header.Add(mynahSettings.AuthSettings.JwtHeader, jwt)

	//create a recorder for the response
	rr := httptest.NewRecorder()

	//make the request
	router.ServeHTTP(rr, req)

	//check the result
	if stat := rr.Code; stat != http.StatusOK {
		t.Fatalf("create user returned non-200: %v want %v", stat, http.StatusOK)
	}

	if rr.Result().ContentLength != int64(len(fileContents)) {
		t.Fatalf("file contents length served (%d) not the same as saved (%d)\n",
			rr.Result().ContentLength,
			len(fileContents))
	}

	if rr.Result().Header.Get("Content-Type") != expectedType {
		t.Fatalf("file contents served (%s) not the same as saved (%s)\n",
			rr.Result().Header.Get("Content-Type"),
			expectedType)
	}

	//delete the file
	if deleteErr := storageProvider.DeleteFile(&file); deleteErr != nil {
		t.Fatalf("failed to delete file: %s", deleteErr)
	}
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
