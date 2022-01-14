package python

import (
	"github.com/go-python/cpy3"
)

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

//manages a loaded python3.7 module
type python3_7Module struct {
  module *python3.PyObject
  //functions loaded in this module
  functions map[string]*python3.PyObject
}

//manages python3.7 interop
type python3_7 struct {
  //the modules that have been loaded
  modules map[string]python3_7Module
}

//group python objects to be dereferenced
type refCountGroup struct {
  objects []*python3.PyObject
}
