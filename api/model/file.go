// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import "strconv"

// FileLocation the permissions a user can have for a project
type FileLocation string

// MynahFileTag a tag on a file version
type MynahFileTag string

const (
	TagLatest   MynahFileTag = "latest"
	TagOriginal MynahFileTag = "original"
)

// MetadataKey a key into file metadata
type MetadataKey string

const (
	MetadataSize     MetadataKey = "size"
	MetadataWidth    MetadataKey = "width"
	MetadataHeight   MetadataKey = "height"
	MetadataChannels MetadataKey = "channels"
)

// FileMetadata metadata type
type FileMetadata map[MetadataKey]string

// MynahFileVersion a version of the file
type MynahFileVersion struct {
	//whether the file version is available locally
	ExistsLocally bool
	//file metadata
	Metadata FileMetadata `json:"metadata" xorm:"TEXT 'metadata'"`
}

// MynahFile Defines a file managed by Mynah
type MynahFile struct {
	//the id of the file
	Uuid string `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the organization this file belongs to
	OrgId string `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//the owner of the file by uuid (allowed to add and remove the file from projects)
	OwnerUuid string `json:"owner_uuid" xorm:"TEXT not null 'owner_uuid'"`
	//the name of the file
	Name string `json:"name" xorm:"TEXT 'name'"`
	//the time the file was uploaded
	Created int64 `json:"-" xorm:"INTEGER 'last_modified'"`
	//the http detected content type (original)
	DetectedContentType string `json:"-" xorm:"TEXT 'detected_content_type'"`
	//versions of the file
	Versions map[MynahFileTag]MynahFileVersion `json:"versions" xorm:"TEXT 'versions'"`
}

// GetDefaultInt GetDefault returns a value if the key is found or the default value provided
func (m FileMetadata) GetDefaultInt(key MetadataKey, def int64) int64 {
	if val, found := m[key]; found {
		if intVal, err := strconv.ParseInt(val, 0, 8); err == nil {
			return intVal
		}
	}
	return def
}
