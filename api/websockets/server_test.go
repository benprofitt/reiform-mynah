package websockets

import (
	"fmt"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"net/http"
	"net/url"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"testing"
	"time"
)

//make requests to the testing server
func testHarnessE2E(path string,
	jwt string,
	uuid *string,
	mynahSettings *settings.MynahSettings,
	wsProvider WebSocketProvider) error {

	//make a request
	u := url.URL{
		Scheme: "ws",
		Host:   "localhost:8080",
		Path:   path,
	}

	headers := make(http.Header)
	headers[mynahSettings.AuthSettings.JwtHeader] = []string{jwt}

	//connect to the websocket server, attaching jwt as auth
	c, _, err := websocket.DefaultDialer.Dial(u.String(), headers)
	if err != nil {
		return err
	}
	defer c.Close()

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
		message := fmt.Sprintf(m)
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
	if authPErr != nil {
		t.Errorf("failed to create auth provider for test %s", authPErr)
		return
	}
	defer authProvider.Close()
	dbProvider, dbPErr := db.NewDBProvider(mynahSettings, authProvider)
	if dbPErr != nil {
		t.Errorf("failed to create database provider for test %s", dbPErr)
		return
	}
	defer dbProvider.Close()

	//create a user for authenticating the request
	admin := model.MynahUser{
		IsAdmin: true,
		OrgId:   uuid.New().String(),
	}

	//create a user
	user, jwt, userErr := authProvider.CreateUser()
	if userErr != nil {
		t.Errorf("error creating user: %s", userErr)
		return
	}

	//create the user in the database
	if createErr := dbProvider.CreateUser(user, &admin); createErr != nil {
		t.Errorf("failed to create test user %s", createErr)
		return
	}

	//create the websocket provider
	wsProvider := NewWebSocketProvider(mynahSettings)
	defer wsProvider.Close()

	//create the middleware router for handling requests
	router := middleware.NewRouter(mynahSettings, authProvider, dbProvider)

	//register the websocket endpoint
	router.HandleHTTPRequest("test", wsProvider.ServerHandler())

	//start the websocket server in a goroutine
	go func() {
		router.ListenAndServe()
	}()

	//let the server come up
	time.Sleep(time.Second * 1)

	//execute the tests on the server
	if err := testHarnessE2E("/api/v1/test", jwt, &user.Uuid, mynahSettings, wsProvider); err != nil {
		t.Errorf("ws server test harness failed %s", err)
	}

	//shut the test server down
	router.Close()
}
