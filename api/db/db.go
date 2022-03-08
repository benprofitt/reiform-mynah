// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"reiform.com/mynah/auth"
	"reiform.com/mynah/log"
	"reiform.com/mynah/settings"
)

// NewDBProvider Create a new db provider based on the Mynah settings
func NewDBProvider(mynahSettings *settings.MynahSettings, authProvider auth.AuthProvider) (DBProvider, error) {
	//check the db settings type
	if mynahSettings.DBSettings.Type == settings.Local {
		return newLocalDB(mynahSettings, authProvider)
	} else if mynahSettings.DBSettings.Type == settings.External {
		log.Fatalf("external db type not implemented")
		return nil, nil
	} else {
		log.Fatalf("invalid database configuration type %s", mynahSettings.DBSettings.Type)
		return nil, nil
	}
}
