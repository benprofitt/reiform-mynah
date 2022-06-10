// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

import (
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"testing"
	"time"
)

//stores a message from python
type testRes struct {
	uuid model.MynahUuid
	msg  []byte
}

//serializes to json for request
type PyReq struct {
	Msg string `json:"msg"`
}

//expected python response
type PyRes struct {
	Msg string `json:"msg"`
}

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

func TestIPC(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.ModulePath = "../../python"
	// without this we seem to get intermittent failures where the domain socket file is not found
	// this may be due to test collisions
	mynahSettings.IPCSettings.SocketAddr = "/tmp/test_mynah.sock"

	//create the ipc provider
	ipcProvider, ipcErr := NewIPCProvider(mynahSettings)
	require.NoError(t, ipcErr)
	defer ipcProvider.Close()

	//create a python provider
	p := python.NewPythonProvider(mynahSettings)
	defer p.Close()

	pyFunction, err := p.InitFunction("mynah_test", "ipc_test")
	require.NoError(t, err)

	messagesToSend := 50

	//capture results
	resChan := make(chan testRes, messagesToSend)

	//track messages sent
	sentMessages := make(map[model.MynahUuid]string)

	//handle ipc events
	go ipcProvider.HandleEvents(func(uuid model.MynahUuid, msg []byte) {
		// write to the channel
		resChan <- testRes{
			uuid: uuid,
			msg:  msg,
		}
	})

	time.Sleep(2 * time.Second)

	//call python function
	for i := 0; i < messagesToSend; i++ {
		targetUuid := model.NewMynahUuid()
		targetContents := uuid.New().String()

		sentMessages[targetUuid] = targetContents

		user := model.MynahUser{
			Uuid: targetUuid,
		}

		req := PyReq{
			Msg: targetContents,
		}

		go func() {
			//call the python ipc function
			res := pyFunction.Call(&user, &req)

			var pythonResponse PyRes

			require.NoError(t, res.GetResponse(&pythonResponse))
			require.Equal(t, targetContents, pythonResponse.Msg)
		}()
	}

	for i := 0; i < messagesToSend; i++ {
		res := <-resChan

		targetPayload, ok := sentMessages[res.uuid]
		require.True(t, ok)
		require.Equal(t, targetPayload, string(res.msg))
	}
}
