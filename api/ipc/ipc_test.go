// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

import (
	"github.com/google/uuid"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"testing"
)

//stores a message from python
type testRes struct {
	uuid *string
	msg  []byte
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

	if err := p.InitModule("mynah_test"); err != nil {
		t.Fatalf("failed to init module: %s", err)
	}

	if err := p.InitFunction("mynah_test", "ipc_test"); err != nil {
		t.Fatalf("failed to init function: %s", err)
	}

	messagesToSend := 250

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

		go func() {
			//call the python ipc function
			res, err := p.CallFunction("mynah_test", "ipc_test",
				targetUuid,
				targetContents,
				mynahSettings.IPCSettings.SocketAddr)

			if err != nil {
				t.Errorf("failed to call function: %s", err)
			}

			sentLength := int64(len(targetUuid) + len(targetContents))

			if res.(int64) != sentLength {
				t.Errorf("python result length (%d) != sent length (%d)", res.(int64), sentLength)
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
