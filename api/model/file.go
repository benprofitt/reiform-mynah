// Copyright (c) 2022 by Reiform. All Rights Reserved.

package model

import (
	"fmt"
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

// MynahFileSet defines a set of files
type MynahFileSet map[MynahUuid]*MynahFile

// MynahVersionedFileSet defines a set of file versions
type MynahVersionedFileSet map[MynahUuid]*MynahFileVersion

// MynahFileMetadata defines the metadata a file may have
type MynahFileMetadata struct {
	Size     int64     `json:"size"`
	Width    int64     `json:"width"`
	Height   int64     `json:"height"`
	Channels int64     `json:"channels"`
	Mean     []float64 `json:"mean"`
	StdDev   []float64 `json:"stddev"`
}

// MynahFileVersion a version of the file
type MynahFileVersion struct {
	// the id of this file
	VersionId MynahFileVersionId `json:"version_id"`
	//whether the file version is available locally
	ExistsLocally bool `json:"exists_locally"`
	//file metadata
	Metadata *MynahFileMetadata `json:"metadata"`
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
	// UploadMimeType is the mime type detected on upload as a string
	UploadMimeType string `json:"uploaded_mime_type" xorm:"TEXT 'uploaded_mime_type'"`
	//versions of the file
	Versions map[MynahFileVersionId]*MynahFileVersion `json:"versions" xorm:"TEXT 'versions'"`
}

// NewFile creates a new file
func NewFile(creator *MynahUser) *MynahFile {
	f := MynahFile{
		Uuid:           NewMynahUuid(),
		OrgId:          creator.OrgId,
		Permissions:    make(map[MynahUuid]Permissions),
		Name:           "None",
		DateCreated:    time.Now().Unix(),
		UploadMimeType: "None",
		Versions:       make(map[MynahFileVersionId]*MynahFileVersion),
	}
	f.Permissions[creator.Uuid] = Owner
	return &f
}

// CopyTo copies file version data
func (src *MynahFileVersion) CopyTo(dest *MynahFileVersion) {
	dest.ExistsLocally = src.ExistsLocally
	dest.Metadata.Size = src.Metadata.Size
	dest.Metadata.Width = src.Metadata.Width
	dest.Metadata.Height = src.Metadata.Height
	dest.Metadata.Channels = src.Metadata.Channels
	dest.Metadata.Mean = append([]float64(nil), src.Metadata.Mean...)
	dest.Metadata.StdDev = append([]float64(nil), src.Metadata.StdDev...)
}

// NewMynahFileVersion creates a new file version
func NewMynahFileVersion(versionId MynahFileVersionId) *MynahFileVersion {
	return &MynahFileVersion{
		VersionId: versionId,
		Metadata: &MynahFileMetadata{
			Size:     0,
			Width:    0,
			Height:   0,
			Channels: 0,
			Mean:     make([]float64, 0),
			StdDev:   make([]float64, 0),
		},
		ExistsLocally: false,
	}
}

// GetFileVersion returns a given version of the file if it exists
func (f *MynahFile) GetFileVersion(v MynahFileVersionId) (*MynahFileVersion, error) {
	if version, ok := f.Versions[v]; ok {
		return version, nil
	}
	return nil, fmt.Errorf("file %s does not have %s version", f.Uuid, v)
}

// IsImage returns true if the file appears to be an image
func (p *MynahFile) IsImage() bool {
	switch p.UploadMimeType {
	case "image/png":
		return true
	case "image/jpeg":
		return true
	case "image/tiff":
		return true
	default:
		return false
	}
}

// GetPermissions Get the permissions that a user has on a given file
func (p *MynahFile) GetPermissions(user *MynahUser) Permissions {
	if v, found := p.Permissions[user.Uuid]; found {
		return v
	} else {
		return None
	}
}

// GetLatestVersions gets the latest version of each file in a set
func (s MynahFileSet) GetLatestVersions() (MynahVersionedFileSet, error) {
	res := make(MynahVersionedFileSet)
	for fileId, file := range s {
		if ver, err := file.GetFileVersion(LatestVersionId); err == nil {
			res[fileId] = ver
		} else {
			return nil, err
		}
	}

	return res, nil
}
