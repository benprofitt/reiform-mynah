// Copyright (c) 2023 by Reiform. All Rights Reserved.

package migrations

import (
	"reiform.com/mynah-api/types"
	"xorm.io/xorm"
)

type noop struct {
}

// Migrate performs the migration
func (noop) Migrate(*xorm.Session) error {
	return nil
}

// Id gets the migration id
func (noop) Id() types.MynahUuid {
	return "1"
}

func init() {
	// register noop migration
	registerMigration(&noop{})
}
