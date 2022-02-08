// Copyright (c) 2022 by Reiform. All Rights Reserved.

package test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/ipc"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"
)

//TODO clean generated files

//maintains the context for testing
type TestContext struct {
	Settings          *settings.MynahSettings
	DBProvider        db.DBProvider
	AuthProvider      auth.AuthProvider
	StorageProvider   storage.StorageProvider
	PythonProvider    python.PythonProvider
	IPCProvider       ipc.IPCProvider
	WebSocketProvider websockets.WebSocketProvider
	AsyncProvider     async.AsyncProvider
	Router            *middleware.MynahRouter
}

//load test context and pass to test handler
func WithTestContext(mynahSettings *settings.MynahSettings,
	handler func(t *TestContext) error) error {
	//initialize auth
	authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	if authErr != nil {
		return authErr
	}
	defer authProvider.Close()

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	if dbErr != nil {
		return dbErr
	}
	defer dbProvider.Close()

	//initialize storage
	storageProvider, storageErr := storage.NewStorageProvider(mynahSettings)
	if storageErr != nil {
		return storageErr
	}
	defer storageProvider.Close()

	//initialize python
	pythonProvider := python.NewPythonProvider(mynahSettings)
	defer pythonProvider.Close()

	//initialize websockets
	websocketProvider := websockets.NewWebSocketProvider(mynahSettings)
	defer websocketProvider.Close()

	//initialize async workers
	asyncProvider := async.NewAsyncProvider(mynahSettings, websocketProvider)
	defer asyncProvider.Close()

	//initialize the python ipc server
	ipcProvider, ipcErr := ipc.NewIPCProvider(mynahSettings)
	if ipcErr != nil {
		return ipcErr
	}
	defer ipcProvider.Close()

	//Note: don't call close on router since we never ListenAndServe()

	return handler(&TestContext{
		Settings:          mynahSettings,
		DBProvider:        dbProvider,
		AuthProvider:      authProvider,
		StorageProvider:   storageProvider,
		PythonProvider:    pythonProvider,
		IPCProvider:       ipcProvider,
		WebSocketProvider: websocketProvider,
		AsyncProvider:     asyncProvider,
		Router:            middleware.NewRouter(mynahSettings, authProvider, dbProvider, storageProvider),
	})
}

//create a user and pass to handler
func (t *TestContext) WithCreateUser(isAdmin bool, handler func(*model.MynahUser, string) error) error {
	user, userJwt, userErr := t.AuthProvider.CreateUser()
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
	if dbErr := t.DBProvider.CreateUser(user, &creator); dbErr != nil {
		return fmt.Errorf("failed to create user %s", dbErr)
	}

	err := handler(user, userJwt)

	//delete the user
	if dbErr := t.DBProvider.DeleteUser(&user.Uuid, &creator); dbErr != nil {
		return fmt.Errorf("failed to delete user %s", dbErr)
	}

	return err
}

//create a file and pass to the handler
func (t *TestContext) WithCreateFile(owner *model.MynahUser, contents string, handler func(*model.MynahFile) error) error {
	var file model.MynahFile

	//create a file
	if createErr := t.DBProvider.CreateFile(&file, owner); createErr != nil {
		return fmt.Errorf("failed to create file in database: %s", createErr)
	}

	//create the file in storage
	storeErr := t.StorageProvider.StoreFile(&file, func(f *os.File) error {
		//write contents to the file
		_, err := f.WriteString(contents)
		return err
	})

	if storeErr != nil {
		return fmt.Errorf("failed to write to file")
	}

	//update the file in the database (written to path)
	if updateErr := t.DBProvider.UpdateFile(&file, owner, "path"); updateErr != nil {
		return fmt.Errorf("failed to update file path in database: %s", updateErr)
	}

	//pass to handler
	err := handler(&file)

	if deleteErr := t.StorageProvider.DeleteFile(&file); deleteErr != nil {
		return fmt.Errorf("failed to delete file: %s", deleteErr)
	}

	//clean the file
	if deleteErr := t.DBProvider.DeleteFile(&file.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete file: %s", deleteErr)
	}

	return err
}

//create a project and pass to the handler
func (t *TestContext) WithCreateProject(owner *model.MynahUser, handler func(*model.MynahProject) error) error {
	var project model.MynahProject

	//create a project
	if createErr := t.DBProvider.CreateProject(&project, owner); createErr != nil {
		return fmt.Errorf("failed to create project in database: %s", createErr)
	}

	//pass to handler
	err := handler(&project)

	//clean the project
	if deleteErr := t.DBProvider.DeleteProject(&project.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete project: %s", deleteErr)
	}

	return err
}

//create a dataset and pass to the handler
func (t *TestContext) WithCreateDataset(owner *model.MynahUser, handler func(*model.MynahDataset) error) error {
	var dataset model.MynahDataset

	//create a dataset
	if createErr := t.DBProvider.CreateDataset(&dataset, owner); createErr != nil {
		return fmt.Errorf("failed to create dataset in database: %s", createErr)
	}

	//pass to handler
	err := handler(&dataset)

	//clean the dataset
	if deleteErr := t.DBProvider.DeleteDataset(&dataset.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete dataset: %s", deleteErr)
	}

	return err
}

//create an image classification dataset and pass to the handler
func (t *TestContext) WithCreateICDataset(owner *model.MynahUser, handler func(*model.MynahICDataset) error) error {
	var dataset model.MynahICDataset

	//create a dataset
	if createErr := t.DBProvider.CreateICDataset(&dataset, owner); createErr != nil {
		return fmt.Errorf("failed to create dataset in database: %s", createErr)
	}

	//pass to handler
	err := handler(&dataset)

	//clean the dataset
	if deleteErr := t.DBProvider.DeleteICDataset(&dataset.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete dataset: %s", deleteErr)
	}

	return err
}

//make a request using the mynah router
func (t *TestContext) WithHTTPRequest(req *http.Request, jwt string, handler func(int, *httptest.ResponseRecorder) error) error {
	//create a recorder for the response
	rr := httptest.NewRecorder()

	//add the auth header
	req.Header.Add(t.Settings.AuthSettings.JwtHeader, jwt)

	//make the request
	t.Router.ServeHTTP(rr, req)

	//call the handler
	return handler(rr.Code, rr)
}
