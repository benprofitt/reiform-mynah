// Copyright (c) 2023 by Reiform. All Rights Reserved.

package models

import (
	"fmt"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/types"
)

// MynahUser Defines a mynah user
type MynahUser struct {
	// UserId is the id of the user
	UserId types.MynahUuid `json:"user_id" xorm:"varchar(36) pk not null unique 'user_id'"`
	// NameFirst is the first name of the user
	NameFirst string `json:"name_first" xorm:"TEXT 'name_first'"`
	// NameLast is the last name of the user
	NameLast string `json:"name_last" xorm:"TEXT 'name_last'"`
}

func init() {
	db.RegisterTables(&MynahUser{})
}

// CreateMynahUser creates a new user
func CreateMynahUser(ctx *db.Context, user *MynahUser) error {
	affected, err := ctx.GetEngine().Insert(user)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("user (%s) not created (no records affected)", user.UserId)
	}
	return nil
}
