//go:build !s3
// +build !s3

// Copyright (c) 2022 by Reiform. All Rights Reserved.

package aws

import (
	"errors"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
)

type noS3Extension struct {
}

// NewS3Extension returns an error since the extension
func NewS3Extension(*settings.MynahSettings) (S3Extension, error) {
	log.Infof("s3 storage extension disabled")
	return &noS3Extension{}, nil
}

// PullExternalFile pulls a file from s3
func (s *noS3Extension) PullExternalFile() (*model.MynahFile, error) {
	return nil, errors.New("s3 extension disabled, unable to pull file from s3")
}

// PushLocalFile pushes a local file to an external bucket
func (s *noS3Extension) PushLocalFile(*model.MynahFile) error {
	return errors.New("s3 extension disabled, unable to push file to s3")
}

// Close closes the s3 extension
func (s *noS3Extension) Close() {

}
