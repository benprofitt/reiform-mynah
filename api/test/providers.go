// Copyright (c) 2022 by Reiform. All Rights Reserved.

package test

import (
	"fmt"
	"github.com/google/uuid"
	"github.com/posener/wstest"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/ipc"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/pyimpl"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"
)

type wsRecorder struct {
	httptest.ResponseRecorder
	server net.Conn
}

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

	orgId string
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

	//initialize python
	pythonProvider := python.NewPythonProvider(mynahSettings)
	defer pythonProvider.Close()

	//create the python impl provider
	pyImplProvider := pyimpl.NewPyImplProvider(mynahSettings, pythonProvider)

	//initialize storage
	storageProvider, storageErr := storage.NewStorageProvider(mynahSettings, pyImplProvider)
	if storageErr != nil {
		return storageErr
	}
	defer storageProvider.Close()

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
		orgId:             uuid.NewString(),
	})
}

//create a user and pass to handler
func (t *TestContext) WithCreateUser(isAdmin bool, handler func(*model.MynahUser, string) error) error {
	//create an admin to insert the admin (must have distinct id)
	creator := model.MynahUser{
		OrgId:   t.orgId,
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
	file, err := t.DBProvider.CreateFile(owner, func(f *model.MynahFile) error {
		//create the file in storage
		return t.StorageProvider.StoreFile(f, owner, func(f *os.File) error {
			//write contents to the file
			_, err := f.WriteString(contents)
			return err
		})
	})

	if err != nil {
		return fmt.Errorf("failed to create file in database: %s", err)
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
	//create a file
	err := t.WithCreateFile(owner, "test_contents", func(f *model.MynahFile) error {

		testTarget := "../../docs/test_image.jpg"

		//write an image to the file
		if err := t.StorageProvider.GetStoredFile(f, model.TagLatest, func(path *string) error {
			//remove the placeholder
			if err := os.Remove(*path); err != nil {
				return err
			}

			//check that the file for testing exists
			_, err := os.Stat(testTarget)
			if err != nil {
				return fmt.Errorf("missing test file %s: %s", testTarget, err)
			}

			//open the test file
			source, err := os.Open(testTarget)
			if err != nil {
				return fmt.Errorf("failed to open test file %s: %s", testTarget, err)
			}
			defer func(source *os.File) {
				err := source.Close()
				if err != nil {
					log.Warnf("failed to close test file %s: %s", testTarget, err)
				}
			}(source)

			destination, err := os.Create(*path)
			if err != nil {
				return fmt.Errorf("failed to create target file %s: %s", *path, err)
			}
			defer func(destination *os.File) {
				err := destination.Close()
				if err != nil {
					log.Warnf("failed to close test file %s: %s", *path, err)
				}
			}(destination)

			_, err = io.Copy(destination, source)
			return err

		}); err != nil {
			return fmt.Errorf("failed to overwrite test file with image: %s", err)
		}

		//create a dataset
		dataset, err := t.DBProvider.CreateICDataset(owner, func(d *model.MynahICDataset) {
			d.Files[f.Uuid] = model.MynahICDatasetFile{
				CurrentClass:      "class1",
				OriginalClass:     "class1",
				ConfidenceVectors: make(model.ConfidenceVectors, 0),
			}
		})
		if err != nil {
			return fmt.Errorf("failed to create dataset in database: %s", err)
		}

		//create a project
		project, err := t.DBProvider.CreateICProject(owner, func(p *model.MynahICProject) {
			//add the dataset
			p.Datasets = append(p.Datasets, dataset.Uuid)
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

		//clean the dataset
		if deleteErr := t.DBProvider.DeleteICDataset(&dataset.Uuid, owner); deleteErr != nil {
			return fmt.Errorf("failed to delete dataset: %s", deleteErr)
		}

		return err
	})

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

//expect a websocket message for the given user
func (t *TestContext) WebsocketListener(jwt string, expect int, readyChan chan struct{}, errChan chan error, handler func(string) error) {
	dialer := wstest.NewDialer(t.Router.HttpMiddleware(t.WebSocketProvider.ServerHandler()))

	headers := make(http.Header)
	headers[t.Settings.AuthSettings.JwtHeader] = []string{jwt}

	conn, _, err := dialer.Dial("ws://test/ws", headers)
	if err != nil {
		errChan <- err
		return
	}
	defer conn.Close()

	//server is ready
	close(readyChan)

	var resErr error

	for i := 0; i < expect; i++ {
		_, message, err := conn.ReadMessage()
		if err != nil {
			errChan <- resErr
			return
		}
		if err = handler(string(message)); err != nil {
			errChan <- resErr
			return
		}
	}

	errChan <- resErr
}
