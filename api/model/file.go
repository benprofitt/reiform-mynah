// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

//the permissions a user can have for a project
type FileLocation string

type MetadataKey string

const (
	MetadataSize   MetadataKey = "size"
	MetadataWidth  MetadataKey = "width"
	MetadataHeight MetadataKey = "height"
	MetadataFormat MetadataKey = "format"
)

//metadata type
type FileMetadata map[MetadataKey]string

const (
	Local FileLocation = "local"
	S3    FileLocation = "s3 "
)

//Defines a file managed by Mynah
type MynahFile struct {
	//the id of the file
	Uuid string `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the organization this file belongs to
	OrgId string `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//the owner of the file by uuid (allowed to add and remove the file from projects)
	OwnerUuid string `json:"owner_uuid" xorm:"TEXT not null 'owner_uuid'"`
	//the name of the file
	Name string `json:"name" xorm:"TEXT 'name'"`
	//the location of the file
	Location FileLocation `json:"-" xorm:"TEXT 'location'"`
	//the path to the file
	Path string `json:"-" xorm:"TEXT 'path'"`
	//last modified time
	LastModified int64 `json:"-" xorm:"INTEGER 'last_modified'"`
	//the http detected content type
	DetectedContentType string `json:"-" xorm:"TEXT 'detected_content_type'"`
	//file metadata
	Metadata FileMetadata `json:"metadata" xorm:"TEXT 'metadata'"`
}
