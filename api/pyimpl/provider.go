// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import (
	"reiform.com/mynah/log"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
)

//create a new provider
func NewPyImplProvider(mynahSettings *settings.MynahSettings,
	pythonProvider python.PythonProvider) PyImplProvider {

	log.Infof("using python impl module %s", mynahSettings.PythonSettings.ModuleName)

	return &localImplProvider{
		pythonProvider: pythonProvider,
		moduleName:     mynahSettings.PythonSettings.ModuleName,
	}
}
