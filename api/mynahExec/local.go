// Copyright (c) 2022 by Reiform. All Rights Reserved.

package mynahExec

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
)

type localExecutor struct {
	//the python executable
	executable string
	//the path to the ipc socket
	ipcSock string
}

type localResponse struct {
	res []byte
	err error
}

// NewLocalExecutor creates an executor that runs python code locally
func NewLocalExecutor(mynahSettings *settings.MynahSettings) (MynahExecutor, error) {
	//check that the file exists
	if _, statErr := os.Stat(mynahSettings.PythonSettings.PythonExecutable); errors.Is(statErr, os.ErrNotExist) {
		return nil, fmt.Errorf("python executable %s does not exist", mynahSettings.PythonSettings.PythonExecutable)
	}
	return &localExecutor{
		executable: mynahSettings.PythonSettings.PythonExecutable,
		ipcSock:    mynahSettings.IPCSettings.SocketAddr,
	}, nil
}

// GetAs parses the response into an interface
func (l localResponse) GetAs(i interface{}) error {
	if l.err != nil {
		return l.err
	}

	return json.Unmarshal(l.res, i)
}

// create an error response
func resFromErr(err error) MynahExecutorResponse {
	return &localResponse{
		res: nil,
		err: err,
	}
}

// Call the configured executable
func (l localExecutor) Call(user *model.MynahUser, operation string, data interface{}) MynahExecutorResponse {
	cmd := exec.Command(l.executable, "--operation", operation, "--ipc-socket-path", l.ipcSock, "--uuid", string(user.Uuid)) // #nosec G204

	dataBody, err := json.Marshal(data)
	if err != nil {
		return resFromErr(fmt.Errorf("failed to marshal input data for python process: %s", err))
	}

	// direct python stderr to regular stream
	cmd.Stderr = os.Stderr

	if data != nil && len(dataBody) > 0 {
		stdinBuffer := bytes.Buffer{}
		stdinBuffer.Write(dataBody)
		cmd.Stdin = &stdinBuffer
	}

	stdoutBuffer := bytes.Buffer{}
	cmd.Stdout = &stdoutBuffer

	if err = cmd.Run(); err != nil {
		return resFromErr(fmt.Errorf("python process failed: %s", err))
	}

	return &localResponse{
		res: stdoutBuffer.Bytes(),
		err: nil,
	}
}

func (l localExecutor) Close() {

}
