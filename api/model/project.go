// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// ProjectPermissions the permissions a user can have for a project
type ProjectPermissions int

const (
	None ProjectPermissions = iota
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
	UserPermissions map[string]ProjectPermissions `json:"-" xorm:"TEXT 'user_permissions'"`
	//the name of the project
	ProjectName string `json:"project_name" xorm:"TEXT 'project_name'"`
	//datasets that are part of this project (by id)
	Datasets []string `json:"datasets" xorm:"TEXT 'datasets'"`
}

//get the base dataset for attributes
func (d *MynahProject) GetBaseProject() *MynahProject {
	return d
}

//Get the permissions that a user has on a given project
func (p *MynahProject) GetPermissions(user *MynahUser) ProjectPermissions {
	if v, found := p.UserPermissions[user.Uuid]; found {
		return v
	} else {
		return None
	}
}
