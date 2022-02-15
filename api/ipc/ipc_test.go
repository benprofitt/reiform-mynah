// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

import (
	"github.com/google/uuid"
	"os"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"testing"
)

//stores a message from python
type testRes struct {
	uuid *string
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

	//create the ipc provider
	ipcProvider, ipcErr := NewIPCProvider(mynahSettings)
	if ipcErr != nil {
		t.Fatalf("failed to init ipc provider: %s", ipcErr)
	}

	defer ipcProvider.Close()

	//create a python provider
	p := python.NewPythonProvider(mynahSettings)
	defer p.Close()

	pyFunction, err := p.InitFunction("mynah_test", "ipc_test")

	if err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	messagesToSend := 50

	//capture results
	resChan := make(chan testRes, messagesToSend)

	//track messages sent
	sentMessages := make(map[string]string)

	//handle ipc events
	go ipcProvider.HandleEvents(func(uuid *string, msg []byte) {
		// write to the channel
		resChan <- testRes{
			uuid: uuid,
			msg:  msg,
		}
	})

	//call python function
	for i := 0; i < messagesToSend; i++ {
		targetUuid := uuid.New().String()
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
			if err := res.GetResponse(&pythonResponse); err != nil {
				t.Errorf("error calling function: %s", err)
			} else {

				if pythonResponse.Msg != targetContents {
					t.Errorf("response msg (%s) did not match target contents (%s)",
						pythonResponse.Msg,
						targetContents)
				}
			}
		}()
	}

	for i := 0; i < messagesToSend; i++ {
		res := <-resChan

		if targetPayload, ok := sentMessages[*res.uuid]; ok {
			//check the message as a string
			if string(res.msg) != targetPayload {
				t.Fatalf("ipc message contents does not match (%s != %s)", string(res.msg), targetPayload)
			}

		} else {
			t.Fatalf("got unexpected uuid: %s", *res.uuid)
		}
	}
}
