// Copyright (c) 2022 by Reiform. All Rights Reserved.

package python

import (
	"reiform.com/mynah/settings"
)

//Create a new python provider
func NewPythonProvider(mynahSettings *settings.MynahSettings) PythonProvider {
	//for now only supports local
	return newLocalPythonProvider(mynahSettings)
}
