// Copyright (c) 2022 by Reiform. All Rights Reserved.

package python

import (
	"reiform.com/mynah/model"
)

// MynahPythonResponse interface for types returned from python calls
//Allows us to decouple execution and results (i.e. over a network)
type MynahPythonResponse interface {
	// GetResponse get the contents of the response
	GetResponse(interface{}) error
}

// MynahPythonFunction call a mynah function
type MynahPythonFunction interface {
	// Call call the function as a user
	Call(*model.MynahUser, interface{}) MynahPythonResponse
}

// PythonProvider defines calling interface for python scripts (or externally)
type PythonProvider interface {
	// InitFunction initialize a module function. First arg is module, second is function name
	InitFunction(string, string) (MynahPythonFunction, error)
	// Close close python
	Close()
}
