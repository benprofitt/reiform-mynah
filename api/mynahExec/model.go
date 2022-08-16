// Copyright (c) 2022 by Reiform. All Rights Reserved.

package mynahExec

import "reiform.com/mynah/model"

type MynahExecutorResponse interface {
	// GetAs parses the response into an interface
	GetAs(interface{}) error
}

type MynahExecutor interface {
	// Call some implementation with the given user, the operation name, and the data to pass
	Call(*model.MynahUser, string, interface{}) MynahExecutorResponse
	// Close the executor
	Close()
}
