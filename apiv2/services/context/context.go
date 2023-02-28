// Copyright (c) 2023 by Reiform. All Rights Reserved.

package context

import "reiform.com/mynah-api/models"

// Context defines application context
type Context struct {
	User *models.MynahUser
}
