// Copyright (c) 2022 by Reiform. All Rights Reserved.

package auth

import (
	"reiform.com/mynah/settings"
)

// NewAuthProvider creates a new auth provider based on the Mynah settings
func NewAuthProvider(mynahSettings *settings.MynahSettings) (AuthProvider, error) {
	//Currently only supports local auth
	return newLocalAuth(mynahSettings)
}
