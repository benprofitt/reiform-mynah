// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import "github.com/google/uuid"

type MynahUuid string

// MynahUser Defines a mynah user
type MynahUser struct {
	//the id of the user
	Uuid MynahUuid `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the id of the organization this user is part of
	OrgId MynahUuid `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//the first name of the user
	NameFirst string `json:"name_first" xorm:"TEXT 'name_first'"`
	//the last name of the user
	NameLast string `json:"name_last" xorm:"TEXT 'name_last'"`
	//whether the user is an admin
	IsAdmin bool `json:"-" xorm:"TEXT not null 'is_admin'"`
	//who created this user
	CreatedBy MynahUuid `json:"-" xorm:"TEXT 'created_by'"`
}

// NewUser creates a user
func NewUser(creator *MynahUser) *MynahUser {
	return &MynahUser{
		Uuid:      NewMynahUuid(),
		OrgId:     creator.OrgId,
		NameFirst: "first",
		NameLast:  "last",
		IsAdmin:   false,
		CreatedBy: creator.Uuid,
	}
}

// NewMynahUuid creates a new mynah uuid
func NewMynahUuid() MynahUuid {
	return MynahUuid(uuid.NewString())
}
