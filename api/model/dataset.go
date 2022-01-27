// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

//Defines a mynah dataset
type MynahDataset struct {
	//the id of the dataset
	Uuid string `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the id of the organization this dataset is part of
	OrgId string `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//the owner
	OwnerUuid string `json:"owner_uuid" xorm:"TEXT not null 'owner_uuid'"`
	//all referenced MynahFiles
	ReferencedFiles []string `json:"referenced_files" xorm:"TEXT 'referenced_files'"`
	//the name of the dataset
	DatasetName string `json:"dataset_name" xorm:"TEXT 'dataset_name'"`
}
