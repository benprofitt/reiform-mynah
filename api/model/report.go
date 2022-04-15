// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import "github.com/google/uuid"

// MynahAbstractReport mynah abstract type
type MynahAbstractReport interface {
	GetBaseReport() *MynahReport
}

// MynahReport a mynah operation report
type MynahReport struct {
	//the id of the report
	Uuid string `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the id of the organization this report is part of
	OrgId string `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//permissions that various users have
	UserPermissions map[string]Permissions `json:"-" xorm:"TEXT 'user_permissions'"`
}

// GetBaseReport get the base dataset for attributes
func (d *MynahReport) GetBaseReport() *MynahReport {
	return d
}

// GetPermissions Get the permissions that a user has on a given dataset
func (p *MynahReport) GetPermissions(user *MynahUser) Permissions {
	if v, found := p.UserPermissions[user.Uuid]; found {
		return v
	} else {
		return None
	}
}

// NewReport creates a new report
func NewReport(creator *MynahUser) *MynahReport {
	return &MynahReport{
		Uuid:            uuid.NewString(),
		OrgId:           creator.OrgId,
		UserPermissions: make(map[string]Permissions),
	}
}
