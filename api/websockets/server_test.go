// Copyright (c) 2022 by Reiform. All Rights Reserved.

package websockets

import (
	"fmt"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/url"
	"os"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"testing"
	"time"
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

//make requests to the testing server
func testHarnessE2E(path string,
	jwt string,
	uuid model.MynahUuid,
	mynahSettings *settings.MynahSettings,
	wsProvider WebSocketProvider) error {

	//make a request
	u := url.URL{
		Scheme: "ws",
		Host:   "localhost:8080",
		Path:   path,
	}

	headers := make(http.Header)
	headers.Add(mynahSettings.AuthSettings.JwtHeader, jwt)

	//connect to the websocket server, attaching jwt as auth
	c, _, err := websocket.DefaultDialer.Dial(u.String(), headers)
	if err != nil {
		return err
	}
	defer func(c *websocket.Conn) {
		err := c.Close()
		if err != nil {
			log.Warnf("failed to close websocket connection: %s", err)
		}
	}(c)

	if err := c.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
		return err
	}

	messagesToSend := 100

	doneChan := make(chan struct{})
	results := make([]string, 0)

	var clientErr error

	//accept responses from the server
	go func() {
		defer close(doneChan)
		for i := 0; i < messagesToSend; i++ {
			_, message, err := c.ReadMessage()
			if err != nil {
				clientErr = err
				return
			}
			results = append(results, string(message))
		}
	}()

	time.Sleep(1 * time.Second)

	for i := 0; i < messagesToSend; i++ {
		//distribute messages
		wsProvider.Send(uuid, []byte(fmt.Sprintf("message-%d", i)))
	}

	//wait for all messages to arrive
	<-doneChan

	if clientErr != nil {
		return clientErr
	}

	//check that the results match
	for i, m := range results {
		message := fmt.Sprint(m)
		expected := fmt.Sprintf("message-%d", i)
		if message != expected {
			return fmt.Errorf("message %s doesn't match expected %s", message, expected)
		}
	}

	return nil
}

//start the testing server
func TestWSServerE2E(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//init the db provider and the auth provider for authenticating the request
	authProvider, authPErr := auth.NewAuthProvider(mynahSettings)
	require.NoError(t, authPErr)
	defer authProvider.Close()

	dbProvider, dbPErr := db.NewDBProvider(mynahSettings, authProvider)
	require.NoError(t, dbPErr)
	defer dbProvider.Close()

	//initialize storage
	storageProvider, storageErr := storage.NewStorageProvider(mynahSettings)
	require.NoError(t, storageErr)
	defer storageProvider.Close()

	admin := model.MynahUser{
		IsAdmin: true,
		OrgId:   model.NewMynahUuid(),
	}

	user, err := dbProvider.CreateUser(&admin, func(user *model.MynahUser) error { return nil })
	require.NoError(t, err)

	//create a user
	jwt, err := authProvider.GetUserAuth(user)
	require.NoError(t, err)

	//create the websocket provider
	wsProvider := NewWebSocketProvider(mynahSettings)
	defer wsProvider.Close()

	//create the middleware router for handling requests
	router := middleware.NewRouter(mynahSettings, authProvider, dbProvider, storageProvider)

	//register the websocket endpoint
	router.HandleHTTPRequest("GET", "integration", wsProvider.ServerHandler())

	//start the websocket server in a goroutine
	go func() {
		router.ListenAndServe()
	}()

	//let the server come up
	time.Sleep(time.Second * 2)

	//execute the tests on the server
	require.NoError(t, testHarnessE2E("/api/v1/integration", jwt, user.Uuid, mynahSettings, wsProvider))

	//shut the integration server down
	router.Close()
}
