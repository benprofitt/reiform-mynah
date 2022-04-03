// Copyright (c) 2022 by Reiform. All Rights Reserved.

package aws

import "reiform.com/mynah/model"

// S3Extension describes the behavior of the s3 extension
type S3Extension interface {
	// PullExternalFile pulls a file from s3 and tracks it in local storage
	PullExternalFile() (*model.MynahFile, error)
	// PushLocalFile pushes a local file to an external bucket
	PushLocalFile(*model.MynahFile) error
	// Close closes the s3 extension
	Close()
}
