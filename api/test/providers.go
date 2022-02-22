// Copyright (c) 2022 by Reiform. All Rights Reserved.

package test

import (
	"fmt"
	"github.com/google/uuid"
	"net/http"
	"net/http/httptest"
	"os"
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/ipc"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/pyimpl"
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
	PyImplProvider    pyimpl.PyImplProvider
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

	//create the python impl provider
	pyImplProvider := pyimpl.NewPyImplProvider(mynahSettings, pythonProvider)

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
		PyImplProvider:    pyImplProvider,
		IPCProvider:       ipcProvider,
		WebSocketProvider: websocketProvider,
		AsyncProvider:     asyncProvider,
		Router:            middleware.NewRouter(mynahSettings, authProvider, dbProvider, storageProvider),
	})
}

//create a user and pass to handler
func (t *TestContext) WithCreateUser(isAdmin bool, handler func(*model.MynahUser, string) error) error {
	//create an admin to insert the admin (must have distinct id)
	creator := model.MynahUser{
		OrgId:   uuid.NewString(),
		IsAdmin: true,
	}

	user, err := t.DBProvider.CreateUser(&creator, func(user *model.MynahUser) {
		user.IsAdmin = isAdmin
	})
	if err != nil {
		return err
	}

	userJwt, err := t.AuthProvider.GetUserAuth(user)
	if err != nil {
		return err
	}

	err = handler(user, userJwt)

	//delete the user
	if dbErr := t.DBProvider.DeleteUser(&user.Uuid, &creator); dbErr != nil {
		return fmt.Errorf("failed to delete user %s", dbErr)
	}

	return err
}

//create a file and pass to the handler
func (t *TestContext) WithCreateFile(owner *model.MynahUser, contents string, handler func(*model.MynahFile) error) error {

	//create a new file
	file, err := t.DBProvider.CreateFile(owner, func(*model.MynahFile) {})

	if err != nil {
		return fmt.Errorf("failed to create file in database: %s", err)
	}

	//create the file in storage
	storeErr := t.StorageProvider.StoreFile(file, func(f *os.File) error {
		//write contents to the file
		_, err := f.WriteString(contents)
		return err
	})

	if storeErr != nil {
		return fmt.Errorf("failed to write to file")
	}

	//pass to handler
	err = handler(file)

	if deleteErr := t.StorageProvider.DeleteFile(file); deleteErr != nil {
		return fmt.Errorf("failed to delete file: %s", deleteErr)
	}

	//clean the file
	if deleteErr := t.DBProvider.DeleteFile(&file.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete file: %s", deleteErr)
	}

	return err
}

//create an icdataset and pass to the handler
func (t *TestContext) WithCreateICDataset(owner *model.MynahUser, handler func(*model.MynahICDataset) error) error {

	dataset, err := t.DBProvider.CreateICDataset(owner, func(*model.MynahICDataset) {})
	if err != nil {
		return fmt.Errorf("failed to create dataset in database: %s", err)
	}

	//pass to handler
	err = handler(dataset)

	//clean the dataset
	if deleteErr := t.DBProvider.DeleteICDataset(&dataset.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete dataset: %s", deleteErr)
	}

	return err
}

//create a project and pass to the handler
func (t *TestContext) WithCreateProject(owner *model.MynahUser, handler func(*model.MynahProject) error) error {
	project, err := t.DBProvider.CreateProject(owner, func(*model.MynahProject) {})
	if err != nil {
		return fmt.Errorf("failed to create project in database: %s", err)
	}

	//pass to handler
	err = handler(project)

	//clean the project
	if deleteErr := t.DBProvider.DeleteProject(&project.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete project: %s", deleteErr)
	}

	return err
}

//create a project and pass to the handler
func (t *TestContext) WithCreateICProject(owner *model.MynahUser, handler func(*model.MynahICProject) error) error {
	project, err := t.DBProvider.CreateICProject(owner, func(*model.MynahICProject) {})
	if err != nil {
		return fmt.Errorf("failed to create project in database: %s", err)
	}

	//pass to handler
	err = handler(project)

	//clean the project
	if deleteErr := t.DBProvider.DeleteICProject(&project.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete project: %s", deleteErr)
	}

	return err
}

//create a dataset and pass to the handler
func (t *TestContext) WithCreateDataset(owner *model.MynahUser, handler func(*model.MynahDataset) error) error {
	dataset, err := t.DBProvider.CreateDataset(owner, func(*model.MynahDataset) {})
	if err != nil {
		return fmt.Errorf("failed to create project in database: %s", err)
	}

	//pass to handler
	err = handler(dataset)

	//clean the dataset
	if deleteErr := t.DBProvider.DeleteDataset(&dataset.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete dataset: %s", deleteErr)
	}

	return err
}

//create a complete ic project with a dataset and file and pass to the handler
func (t *TestContext) WithCreateFullICProject(owner *model.MynahUser, handler func(*model.MynahICProject) error) error {
	//create a dataset
	dataset, err := t.DBProvider.CreateICDataset(owner, func(d *model.MynahICDataset) {
		d.Classes = append(d.Classes, "test_class")
	})
	if err != nil {
		return fmt.Errorf("failed to create dataset in database: %s", err)
	}

	//create a file
	err = t.WithCreateFile(owner, "test_contents", func(f *model.MynahFile) error {

		//create a project
		project, err := t.DBProvider.CreateICProject(owner, func(p *model.MynahICProject) {
			//add the dataset
			p.DatasetAttributes[dataset.Uuid] = model.MynahICProjectData{
				Data: make(map[string]map[string]model.MynahICProjectClassFileData),
			}

			p.DatasetAttributes[dataset.Uuid].Data["test_class"] = make(map[string]model.MynahICProjectClassFileData)

			p.DatasetAttributes[dataset.Uuid].Data["test_class"][f.Uuid] = model.MynahICProjectClassFileData{
				CurrentClass: "test_class",
				OriginalClass: "old_class",
				ConfidenceVectors: make([][]float64, 0),
			}
		})

		if err != nil {
			return fmt.Errorf("failed to create project in database: %s", err)
		}

		//pass to handler
		err = handler(project)

		//clean the project
		if deleteErr := t.DBProvider.DeleteICProject(&project.Uuid, owner); deleteErr != nil {
			return fmt.Errorf("failed to delete project: %s", deleteErr)
		}

		return err
	})

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
