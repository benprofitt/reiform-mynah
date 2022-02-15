// Copyright (c) 2022 by Reiform. All Rights Reserved.

package python

import (
	"reiform.com/mynah/model"
)

//interface for types returned from python calls
//Allows us to decouple execution and results (i.e. over a network)
type MynahPythonResponse interface {
	//get the contents of the response
	GetResponse(interface{}) error
}

//call a mynah function
type MynahPythonFunction interface {
	//call the function as a user
	Call(*model.MynahUser, interface{}) MynahPythonResponse
}

//defines calling interface for python scripts (or externally)
type PythonProvider interface {
	//initialize a module function. First arg is module, second is function name
	InitFunction(string, string) (MynahPythonFunction, error)
	//close python
	Close()
}
