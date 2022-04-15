// Copyright (c) 2022 by Reiform. All Rights Reserved.

package test

import (
	"fmt"
	"github.com/gorilla/websocket"
	"github.com/posener/wstest"
	"io"
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
	"reiform.com/mynah/tools"
	"reiform.com/mynah/websockets"
)

// TestContext maintains the context for testing
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

	orgId model.MynahUuid
}

// WithTestContext load test context and pass to test handler
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
		orgId:             model.NewMynahUuid(),
	})
}

// WithCreateUser create a user and pass to handler
func (t *TestContext) WithCreateUser(isAdmin bool, handler func(*model.MynahUser, string) error) error {
	//create an admin to insert the admin (must have distinct id)
	creator := model.MynahUser{
		OrgId:   t.orgId,
		IsAdmin: true,
	}

	user, err := t.DBProvider.CreateUser(&creator, func(user *model.MynahUser) error {
		user.IsAdmin = isAdmin
		return nil
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
	if dbErr := t.DBProvider.DeleteUser(user.Uuid, &creator); dbErr != nil {
		return fmt.Errorf("failed to delete user %s", dbErr)
	}

	return err
}

// WithCreateFile create a file and pass to the handler
func (t *TestContext) WithCreateFile(owner *model.MynahUser, contents string, handler func(*model.MynahFile) error) error {
	//create a new file
	file, err := t.DBProvider.CreateFile(owner, func(mynahFile *model.MynahFile) error {
		//create the file in storage
		return t.StorageProvider.StoreFile(mynahFile, owner, func(f *os.File) error {
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
	if deleteErr := t.DBProvider.DeleteFile(file.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete file: %s", deleteErr)
	}

	return err
}

// WithCreateFiles create a file and pass to the handler
func (t *TestContext) WithCreateFiles(owner *model.MynahUser, fileIds []model.MynahUuid, handler func([]*model.MynahFile) error) error {
	allFiles := make([]*model.MynahFile, 0)

	for _, fileId := range fileIds {
		//create a new file
		file, err := t.DBProvider.CreateFile(owner, func(mynahFile *model.MynahFile) error {
			mynahFile.Uuid = fileId
			//create the file in storage
			return t.StorageProvider.StoreFile(mynahFile, owner, func(f *os.File) error {
				//write contents to the file
				_, err := f.WriteString("(empty)")
				return err
			})
		})
		if err != nil {
			return fmt.Errorf("failed to create file in database: %s", err)
		}

		allFiles = append(allFiles, file)
	}

	//pass to handler
	err := handler(allFiles)

	for _, file := range allFiles {
		if deleteErr := t.StorageProvider.DeleteFile(file); deleteErr != nil {
			return fmt.Errorf("failed to delete file: %s", deleteErr)
		}

		//clean the file
		if deleteErr := t.DBProvider.DeleteFile(file.Uuid, owner); deleteErr != nil {
			return fmt.Errorf("failed to delete file: %s", deleteErr)
		}
	}

	return err
}

// WithCreateICDataset create an icdataset and pass to the handler
func (t *TestContext) WithCreateICDataset(owner *model.MynahUser, handler func(*model.MynahICDataset) error) error {

	dataset, err := t.DBProvider.CreateICDataset(owner, func(*model.MynahICDataset) error { return nil })
	if err != nil {
		return fmt.Errorf("failed to create dataset in database: %s", err)
	}

	//pass to handler
	err = handler(dataset)

	//clean the dataset
	if deleteErr := t.DBProvider.DeleteICDataset(dataset.Uuid, owner); deleteErr != nil {
		return fmt.Errorf("failed to delete dataset: %s", deleteErr)
	}

	return err
}

// WithCreateICDiagnosisReport create a diagnosis report
func (t *TestContext) WithCreateICDiagnosisReport(owner *model.MynahUser, handler func(report *model.MynahICDiagnosisReport) error) error {
	report, err := t.DBProvider.CreateICDiagnosisReport(owner, func(report *model.MynahICDiagnosisReport) error {
		report.ImageData["imageid"] = &model.MynahICDiagnosisReportImageMetadata{
			Class:      "class1",
			Mislabeled: true,
			Point: model.MynahICDiagnosisReportPoint{
				X: 0,
				Y: 0,
			},
			OutlierSets: []string{"lighting"},
		}
		report.ImageData["imageid2"] = &model.MynahICDiagnosisReportImageMetadata{
			Class:      "class2",
			Mislabeled: false,
			Point: model.MynahICDiagnosisReportPoint{
				X: 0,
				Y: 0,
			},
			OutlierSets: []string{},
		}
		report.Breakdown["class1"] = &model.MynahICDiagnosisReportBucket{}
		report.Breakdown["class2"] = &model.MynahICDiagnosisReportBucket{}
		return nil
	})
	if err != nil {
		return fmt.Errorf("failed to create report in database: %s", err)
	}

	//pass to handler
	err = handler(report)

	//TODO clean report?

	return err
}

// WithCreateFullICDataset create a complete ic dataset
func (t *TestContext) WithCreateFullICDataset(owner *model.MynahUser, withFileIds []model.MynahUuid, handler func(*model.MynahICDataset) error) error {
	//create a file
	err := t.WithCreateFiles(owner, withFileIds, func(files []*model.MynahFile) error {

		testTarget := "../../docs/test_image.jpg"

		for _, f := range files {
			//write an image to the file
			if err := t.StorageProvider.GetStoredFile(f, model.LatestVersionId, func(path *string) error {
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
		}

		//create a dataset
		dataset, err := t.DBProvider.CreateICDataset(owner, func(d *model.MynahICDataset) error {
			if initialVersion, _, err := tools.MakeICDatasetVersion(d, owner, t.StorageProvider, t.DBProvider); err == nil {
				for _, f := range files {
					initialVersion.Files[f.Uuid] = &model.MynahICDatasetFile{
						ImageVersionId:    model.LatestVersionId,
						CurrentClass:      "class1",
						OriginalClass:     "class1",
						ConfidenceVectors: make(model.ConfidenceVectors, 0),
						Projections:       make(map[string][]int),
					}
				}
			} else {
				return err
			}
			return nil
		})
		if err != nil {
			return fmt.Errorf("failed to create dataset in database: %s", err)
		}

		//pass to handler
		err = handler(dataset)

		//clean the dataset
		if deleteErr := t.DBProvider.DeleteICDataset(dataset.Uuid, owner); deleteErr != nil {
			return fmt.Errorf("failed to delete dataset: %s", deleteErr)
		}

		return err
	})

	return err
}

// WithHTTPRequest make a request using the mynah router
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

// WebsocketListener expect a websocket message for the given user
func (t *TestContext) WebsocketListener(jwt string, expect int, readyChan chan struct{}, errChan chan error, handler func([]byte) error) {
	dialer := wstest.NewDialer(t.Router.HttpMiddleware(t.WebSocketProvider.ServerHandler()))

	headers := make(http.Header)
	headers[t.Settings.AuthSettings.JwtHeader] = []string{jwt}

	conn, _, err := dialer.Dial("ws://test/ws", headers)
	if err != nil {
		errChan <- err
		return
	}
	defer func(conn *websocket.Conn) {
		err := conn.Close()
		if err != nil {
			log.Warnf("error closing websocket connection: %s", err)
		}
	}(conn)

	//server is ready
	close(readyChan)

	for i := 0; i < expect; i++ {
		_, message, err := conn.ReadMessage()
		if err != nil {
			errChan <- err
			return
		}
		if err = handler(message); err != nil {
			errChan <- err
			return
		}
	}

	close(errChan)
}
