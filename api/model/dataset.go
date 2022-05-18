// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
	"time"
)

// MynahAbstractDataset mynah abstract type
type MynahAbstractDataset interface {
	GetBaseDataset() *MynahDataset
}

// Permissions the permissions a user can have for a dataset
type Permissions int

const (
	None Permissions = iota
	Read
	Edit
	Owner
)

// MynahDatasetVersionId a version id for a dataset
type MynahDatasetVersionId string

// MynahClassName is used for class names
type MynahClassName string

// MynahDataset Defines a mynah dataset
type MynahDataset struct {
	//the id of the dataset
	Uuid MynahUuid `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the id of the organization this dataset is part of
	OrgId MynahUuid `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//permissions for users
	Permissions map[MynahUuid]Permissions `json:"permissions" xorm:"TEXT 'permissions'"`
	//the name of the dataset
	DatasetName string `json:"dataset_name" xorm:"TEXT 'dataset_name'"`
	//the date created as a unix timestamp
	DateCreated int64 `json:"date_created" xorm:"INTEGER 'date_created'"`
	//the date modified as a unix timestamp
	DateModified int64 `json:"date_modified" xorm:"INTEGER 'date_modified'"`
}

// GetBaseDataset get the base dataset for attributes
func (d *MynahDataset) GetBaseDataset() *MynahDataset {
	return d
}

// GetPermissions Get the permissions that a user has on a given dataset
func (p *MynahDataset) GetPermissions(user *MynahUser) Permissions {
	if v, found := p.Permissions[user.Uuid]; found {
		return v
	} else {
		return None
	}
}

// NewDataset creates a new dataset
func NewDataset(creator *MynahUser) *MynahDataset {
	dataset := MynahDataset{
		Uuid:         NewMynahUuid(),
		OrgId:        creator.OrgId,
		Permissions:  make(map[MynahUuid]Permissions),
		DatasetName:  "no name",
		DateCreated:  time.Now().Unix(),
		DateModified: time.Now().Unix(),
	}

	dataset.Permissions[creator.Uuid] = Owner
	return &dataset
}
