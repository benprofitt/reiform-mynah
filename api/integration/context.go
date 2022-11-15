// Copyright (c) 2022 by Reiform. All Rights Reserved.

package integration

import (
	"database/sql"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reiform.com/mynah/api"
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/impl"
	"reiform.com/mynah/ipc"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/mynahExec"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"

	"github.com/go-testfixtures/testfixtures/v3"
	_ "github.com/mattn/go-sqlite3"
)

// Context maintains the context for testing
type Context struct {
	Settings          *settings.MynahSettings
	DBProvider        db.DBProvider
	AuthProvider      auth.AuthProvider
	StorageProvider   storage.StorageProvider
	ImplProvider      impl.ImplProvider
	IPCServer         ipc.IPCServer
	WebSocketProvider websockets.WebSocketProvider
	AsyncProvider     async.AsyncProvider
	Router            *middleware.MynahRouter
}

// WithTestContext load integration context and pass to integration handler
func WithTestContext(mynahSettings *settings.MynahSettings,
	fixturesPath string,
	handler func(*Context)) error {
	//initialize auth
	authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	if authErr != nil {
		return authErr
	}
	defer authProvider.Close()

	// use a integration database
	mynahSettings.DBSettings.LocalPath = "data/test.db"
	mynahSettings.DBSettings.Type = "local"

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	if dbErr != nil {
		return dbErr
	}
	defer dbProvider.Close()

	//open a connection to the db to load
	testDB, err := sql.Open("sqlite3", mynahSettings.DBSettings.LocalPath)
	if err != nil {
		return fmt.Errorf("failed to connect to database: %s", err)
	}

	// open integration fixtures
	fixtures, err := testfixtures.New(testfixtures.Database(testDB),
		testfixtures.Dialect("sqlite"),
		testfixtures.Directory(fixturesPath))
	if err != nil {
		return fmt.Errorf("failed to init integration fixtures: %s", err)
	}

	if err = fixtures.Load(); err != nil {
		return fmt.Errorf("failed to load integration fixtures: %s", err)
	}

	//initialize storage
	storageProvider, storageErr := storage.NewStorageProvider(mynahSettings)
	if storageErr != nil {
		return storageErr
	}
	defer storageProvider.Close()

	executor, err := mynahExec.NewLocalExecutor(mynahSettings)
	if err != nil {
		return err
	}
	defer executor.Close()

	//create the python impl provider
	implProvider := impl.NewImplProvider(mynahSettings, dbProvider, storageProvider, executor)

	//initialize websockets
	websocketProvider := websockets.NewWebSocketProvider(mynahSettings)
	defer websocketProvider.Close()

	//initialize async workers
	asyncProvider := async.NewAsyncProvider(mynahSettings, websocketProvider)
	defer asyncProvider.Close()

	//initialize the python ipc server
	icpServer, ipcErr := ipc.NewIPCServer(mynahSettings.IPCSettings.SocketAddr)
	if ipcErr != nil {
		return ipcErr
	}
	defer icpServer.Close()

	//Note: don't call close on router since we never ListenAndServe()
	c := Context{
		Settings:          mynahSettings,
		DBProvider:        dbProvider,
		AuthProvider:      authProvider,
		StorageProvider:   storageProvider,
		ImplProvider:      implProvider,
		IPCServer:         icpServer,
		WebSocketProvider: websocketProvider,
		AsyncProvider:     asyncProvider,
		Router:            middleware.NewRouter(mynahSettings, authProvider, dbProvider, storageProvider),
	}

	//register routes
	routesErr := api.RegisterRoutes(c.Router,
		c.DBProvider,
		c.AuthProvider,
		c.StorageProvider,
		c.ImplProvider,
		c.WebSocketProvider,
		c.AsyncProvider,
		c.Settings)

	if routesErr != nil {
		return routesErr
	}

	handler(&c)
	return nil
}

// WithHTTPRequest make a request using the mynah router
func (t *Context) WithHTTPRequest(req *http.Request, jwt string, handler func(int, *httptest.ResponseRecorder)) error {
	//create a recorder for the response
	rr := httptest.NewRecorder()

	//add the auth header
	req.Header.Add(t.Settings.AuthSettings.JwtHeader, jwt)

	//make the request
	t.Router.ServeHTTP(rr, req)

	//call the handler
	handler(rr.Code, rr)
	return nil
}
