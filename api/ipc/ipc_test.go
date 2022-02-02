// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

import (
	"github.com/google/uuid"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"testing"
)

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

	doneChan := make(chan struct{})

	targetUuid := uuid.New().String()
	targetContents := "payload"
	var readBytes int64

	//handle ipc events
	go ipcProvider.HandleEvents(func(uuid *string, msg []byte) {
		defer close(doneChan)
		//check the result
		if *uuid != targetUuid {
			t.Errorf("ipc uuid does not match (%s != %s)", *uuid, targetUuid)
		}

		//check the message as a string
		if string(msg) != targetContents {
			t.Errorf("ipc message contents does not match (%s != %s)", string(msg), targetContents)
		}

		readBytes = int64(uuidLength + len(msg))
	})

	res, err := p.CallFunction("mynah_test", "ipc_test",
		targetUuid,
		targetContents,
		mynahSettings.IPCSettings.SocketAddr)

	//call the python ipc function
	if err != nil {
		t.Fatalf("failed to call function: %s", err)
	}

	sentBytes := res.(int64)

	//wait for completion
	<-doneChan

	if readBytes != sentBytes {
		t.Errorf("bytes read (%d) different than bytes written (%d)", readBytes, sentBytes)
	}
}
