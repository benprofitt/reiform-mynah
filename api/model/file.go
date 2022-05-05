// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
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

type FileMetadataValueType interface{}

// FileMetadata metadata type
type FileMetadata map[MetadataKey]FileMetadataValueType

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
	//permissions for users
	Permissions map[MynahUuid]Permissions `json:"permissions" xorm:"TEXT 'permissions'"`
	//the name of the file
	Name string `json:"name" xorm:"TEXT 'name'"`
	//the time the file was uploaded
	DateCreated int64 `json:"date_created" xorm:"INTEGER 'date_created'"`
	//the http detected content type (original)
	DetectedContentType string `json:"-" xorm:"TEXT 'detected_content_type'"`
	//the initial mean
	InitialMean []float64 `json:"-" xorm:"TEXT 'initial_mean'"`
	//the initial stddev
	InitialStdDev []float64 `json:"-" xorm:"TEXT 'initial_std_dev'"`
	//versions of the file
	Versions map[MynahFileVersionId]*MynahFileVersion `json:"versions" xorm:"TEXT 'versions'"`
}

// GetDefaultInt returns a value if the key is found or the default value provided
func (m FileMetadata) GetDefaultInt(key MetadataKey, def int64) int64 {
	if val, found := m[key]; found {
		switch v := val.(type) {
		case int64:
			return v
		case int:
			return int64(v)
		default:
			return def
		}
	}
	return def
}

// GetDefaultFloatSlice returns a value if the key is found or the default value provided
func (m FileMetadata) GetDefaultFloatSlice(key MetadataKey, def []float64) []float64 {
	if val, found := m[key]; found {
		switch v := val.(type) {
		case []float64:
			return v
		case []float32:
			res := make([]float64, len(v))
			for i := 0; i < len(v); i++ {
				res[i] = float64(v[i])
			}
			return res
		default:
			return def
		}
	}
	return def
}

// NewFile creates a new file
func NewFile(creator *MynahUser) *MynahFile {
	f := MynahFile{
		Uuid:                NewMynahUuid(),
		OrgId:               creator.OrgId,
		Permissions:         make(map[MynahUuid]Permissions),
		Name:                "None",
		DateCreated:         time.Now().Unix(),
		DetectedContentType: "None",
		Versions:            make(map[MynahFileVersionId]*MynahFileVersion),
	}
	f.Permissions[creator.Uuid] = Owner
	return &f
}

// GetPermissions Get the permissions that a user has on a given file
func (p *MynahFile) GetPermissions(user *MynahUser) Permissions {
	if v, found := p.Permissions[user.Uuid]; found {
		return v
	} else {
		return None
	}
}
