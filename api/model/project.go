// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// Permissions the permissions a user can have for a project
type Permissions int

const (
	None Permissions = iota
	Read
	Edit
	Owner
)

// MynahAbstractProject mynah abstract type
type MynahAbstractProject interface {
	GetBaseProject() *MynahProject
}

// MynahProject Defines a mynah project
type MynahProject struct {
	//the id of the project
	Uuid string `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the id of the organization this project is part of
	OrgId string `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//permissions that various users have
	UserPermissions map[string]Permissions `json:"-" xorm:"TEXT 'user_permissions'"`
	//the name of the project
	ProjectName string `json:"project_name" xorm:"TEXT 'project_name'"`
	//the date created as a unix timestamp
	DateCreated int64 `json:"date_created" xorm:"INTEGER 'date_created'"`
	//the date modified as a unix timestamp
	DateModified int64 `json:"date_modified" xorm:"INTEGER 'date_modified'"`
}

// GetBaseProject get the base dataset for attributes
func (d *MynahProject) GetBaseProject() *MynahProject {
	return d
}

// GetPermissions Get the permissions that a user has on a given project
func (p *MynahProject) GetPermissions(user *MynahUser) Permissions {
	if v, found := p.UserPermissions[user.Uuid]; found {
		return v
	} else {
		return None
	}
}
