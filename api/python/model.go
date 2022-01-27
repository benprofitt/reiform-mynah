// Copyright (c) 2022 by Reiform. All Rights Reserved.

package python

//defines calling interface for python scripts
type PythonProvider interface {
	//initialize a module by name
	InitModule(string) error
	//initialize a module function. First arg is module, second is function name
	InitFunction(string, string) error
	//call a module function with arguments
	CallFunction(string, string, ...interface{}) (interface{}, error)
	//close python
	Close()
}
