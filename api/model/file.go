// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
	"strconv"
	"time"
)

// FileLocation the location of a file being tracked by Mynah
type FileLocation string

// MynahFileVersionId an id for a given version of the file (either latest, original or a SHA1 hash)
type MynahFileVersionId string

const (
	LatestVersionId   MynahFileVersionId = "latest"
	OriginalVersionId MynahFileVersionId = "original"
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
	ExistsLocally bool `json:"exists_locally"`
	//file metadata
	Metadata FileMetadata `json:"metadata"`
}

// MynahFile Defines a file managed by Mynah
type MynahFile struct {
	//the id of the file
	Uuid MynahUuid `json:"uuid" xorm:"varchar(36) not null unique index 'uuid'"`
	//the organization this file belongs to
	OrgId MynahUuid `json:"-" xorm:"varchar(36) not null 'org_id'"`
	//the owner of the file by uuid
	OwnerUuid MynahUuid `json:"owner_uuid" xorm:"TEXT not null 'owner_uuid'"`
	//the name of the file
	Name string `json:"name" xorm:"TEXT 'name'"`
	//the time the file was uploaded
	DateCreated int64 `json:"date_created" xorm:"INTEGER 'date_created'"`
	//the http detected content type (original)
	DetectedContentType string `json:"-" xorm:"TEXT 'detected_content_type'"`
	//versions of the file
	Versions map[MynahFileVersionId]*MynahFileVersion `json:"versions" xorm:"TEXT 'versions'"`
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

// NewFile creates a new file
func NewFile(creator *MynahUser) *MynahFile {
	return &MynahFile{
		Uuid:                NewMynahUuid(),
		OrgId:               creator.OrgId,
		OwnerUuid:           creator.Uuid,
		Name:                "file_name",
		DateCreated:         time.Now().Unix(),
		DetectedContentType: "none",
		Versions:            make(map[MynahFileVersionId]*MynahFileVersion),
	}
}
