// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

// MynahAbstractDataset mynah abstract type
type MynahAbstractDataset interface {
	GetBaseDataset() *MynahDataset
}

// MynahDataset Defines a mynah dataset
type MynahDataset struct {
	//the id of the dataset
	Uuid string `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the id of the organization this dataset is part of
	OrgId string `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//the owner
	OwnerUuid string `json:"owner_uuid" xorm:"TEXT not null 'owner_uuid'"`
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
