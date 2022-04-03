//go:build s3
// +build s3

// Copyright (c) 2022 by Reiform. All Rights Reserved.

package aws

import (
	"errors"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
)

type s3Extension struct {
}

// NewS3Extension returns an error since the extension
func NewS3Extension(mynahSettings *settings.MynahSettings) (S3Extension, error) {
	log.Infof("s3 storage extension enabled")
	return &s3Extension{}, nil
}

// PullExternalFile pulls a file from s3
func (s *s3Extension) PullExternalFile() (*model.MynahFile, error) {
	return nil, errors.New("s3 extension unimpl")
}

// PushLocalFile pushes a local file to an external bucket
func (s *s3Extension) PushLocalFile(*model.MynahFile) error {
	return errors.New("s3 extension unimpl")
}

// Close closes the s3 extension
func (s *s3Extension) Close() {

}
