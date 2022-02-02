// Copyright (c) 2022 by Reiform. All Rights Reserved.

package python

import (
	"errors"
	"fmt"
	"github.com/go-python/cpy3"
	"log"
	"reflect"
	"reiform.com/mynah/settings"
)

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

//Create a new python provider
func NewPythonProvider(mynahSettings *settings.MynahSettings) PythonProvider {
	python3.Py_Initialize()

	if !python3.Py_IsInitialized() {
		log.Fatalf("error initializing the python interpreter")
	}

	//add src directory to the path so the module can be loaded
	python3.PyList_Append(python3.PySys_GetObject("path"), python3.PyUnicode_FromString(mynahSettings.PythonSettings.ModulePath))

	return &python3_7{
		modules: make(map[string]python3_7Module),
	}
}

//initialize a module by name
func (p *python3_7) InitModule(module string) error {
	if m := python3.PyImport_ImportModule(module); (m != nil) && (python3.PyErr_Occurred() == nil) {
		p.modules[module] = python3_7Module{
			module:    m,
			functions: make(map[string]*python3.PyObject),
		}
		return nil
	} else {
		python3.PyErr_Print()
		return fmt.Errorf("failed to load python module %s", module)
	}
}

//initialize a module function. First arg is module, second is function name
func (p *python3_7) InitFunction(module string, function string) error {
	//verify that the module has been loaded
	if m, found := p.modules[module]; found {
		if fn := m.module.GetAttrString(function); (fn != nil) && (python3.PyErr_Occurred() == nil) {
			m.functions[function] = fn
			return nil
		} else {
			python3.PyErr_Print()
			return fmt.Errorf("failed to load python function %s in module %s", function, module)
		}

	} else {
		return fmt.Errorf("module %s has not been loaded", module)
	}
}

//add a python object to the dereference group
func (g *refCountGroup) append(o *python3.PyObject) *python3.PyObject {
	g.objects = append(g.objects, o)
	return o
}

//dereference python objects in group for python garbage collector
func (g *refCountGroup) decref() {
	for _, o := range g.objects {
		if o != nil {
			o.DecRef()
		}
	}
}

//convert some go type to be a python object
func toPythonObj(value interface{}) (*python3.PyObject, error) {
	v := reflect.ValueOf(value)
	//determine the value of the interface
	switch v.Kind() {
	case reflect.Bool:
		return nil, errors.New("booleans not currently supported for conversion to python objects")

	case reflect.Int, reflect.Int8, reflect.Int32, reflect.Int64:
		return python3.PyLong_FromLongLong(v.Int()), nil

	case reflect.Float32, reflect.Float64:
		return python3.PyFloat_FromDouble(v.Float()), nil

	case reflect.String:
		return python3.PyUnicode_FromString(v.String()), nil

	case reflect.Slice:
		return nil, errors.New("slices not currently supported for conversion to python objects")

		//FIXME can't reflect on slice elements since they aren't interfaces?
		// i.e. change the recursive call

		// s := reflect.ValueOf(v)
		//
		// //create a new python list
		// pyList := g.append(python3.PyList_New(s.Len()))
		//
		// //add each element
		// for i := 0; i < s.Len(); i++ {
		// 	//convert the slice element
		// 	if sliceObj, sliceErr := g.toPythonObj(s.Index(i)); sliceErr == nil {
		// 		//add to the list
		// 		if set := python3.PyList_SetItem(pyList, i, sliceObj); set != 0 {
		// 			return nil, errors.New("failed to add python object to list when converting")
		// 		}
		// 	} else {
		// 		return nil, sliceErr
		// 	}
		// }
		// return pyList, nil

	case reflect.Map:
		return nil, errors.New("maps not currently supported for conversion to python objects")

	default:
		return nil, errors.New("invalid type conversion to python object")
	}
}

//convert some python object to a go type
func (g *refCountGroup) toGoType(obj *python3.PyObject) (interface{}, error) {
	if obj == nil {
		return nil, nil
	}

	//add to reference count group (we now own this object)
	g.append(obj)

	//determine the type of the object
	if python3.PyFloat_Check(obj) {
		return python3.PyFloat_AsDouble(obj), nil

	} else if python3.PyLong_Check(obj) {
		return python3.PyLong_AsLongLong(obj), nil

	} else if python3.PyUnicode_Check(obj) {
		return python3.PyUnicode_AsUTF8(obj), nil

	} else if obj == python3.Py_None {
		return nil, nil

	} else {
		return nil, errors.New("python returned an unsupported type")
	}
}

//call a module function with arguments
func (p *python3_7) CallFunction(module string, function string, args ...interface{}) (interface{}, error) {
	//create a new reference counter group
	var rcg refCountGroup
	defer rcg.decref()

	//locate the module and function
	if m, moduleFound := p.modules[module]; moduleFound {
		if f, functionFound := m.functions[function]; functionFound {
			//create a list for the function arguments
			pyArgs := rcg.append(python3.PyTuple_New(len(args)))
			if (pyArgs == nil) || (python3.PyErr_Occurred() != nil) {
				python3.PyErr_Print()
				return nil, fmt.Errorf("failed to create python args tuple for function %s in module %s", function, module)
			}

			//pack the args
			for i, a := range args {
				//get the type
				//Note: PyTuple_SetItem steals a reference to pyObj
				//so we don't add pyObj to the ref count group
				if pyObj, pyObjErr := toPythonObj(a); pyObjErr == nil {
					python3.PyTuple_SetItem(pyArgs, i, pyObj)
				} else {
					return nil, pyObjErr
				}
			}

			//execute the call
			callRes := f.Call(pyArgs, rcg.append(python3.PyDict_New()))

			if python3.PyErr_Occurred() != nil {
				python3.PyErr_Print()
				return nil, fmt.Errorf("failed to call function %s in module %s", function, module)
			}

			//convert the return type to a go type
			return rcg.toGoType(callRes)

		} else {
			return nil, fmt.Errorf("function %s in module %s not loaded", function, module)
		}
	} else {
		return nil, fmt.Errorf("module %s not loaded", module)
	}
}

//on shutdown
func (p *python3_7) Close() {
	log.Printf("python engine shutdown")
	python3.Py_Finalize()
}
