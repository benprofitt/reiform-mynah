// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahBinObject Defines some binary object stored by the database
type MynahBinObject struct {
	//the id of the data
	Uuid MynahUuid `xorm:"varchar(36) not null unique index 'uuid'"`
	//the id of the organization this data is part of
	OrgId MynahUuid `xorm:"varchar(36) not null 'org_id'"`
	//the binary data
	Data []byte `xorm:"BINARY 'data'"`
}

// NewBinObject creates a new binary data object
func NewBinObject(creator *MynahUser) *MynahBinObject {
	return &MynahBinObject{
		Uuid:  NewMynahUuid(),
		OrgId: creator.OrgId,
		Data:  nil,
	}
}
