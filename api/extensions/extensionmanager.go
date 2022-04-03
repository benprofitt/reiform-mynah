// Copyright (c) 2022 by Reiform. All Rights Reserved.

package extensions

import (
	"reiform.com/mynah/aws"
	"reiform.com/mynah/settings"
)

// NewExtensionManager loads extensions from settings
// These extensions are services that do not _replace_ local behavior. For example, s3 may be supported
// in addition to local storage but not in place of. Alternatively, a different database provider may be used,
// but it will be used _instead_ of sqlite
// see docs/extensions.md
func NewExtensionManager(mynahSettings *settings.MynahSettings) (*ExtensionManager, error) {
	s3Extension, err := aws.NewS3Extension(mynahSettings)
	if err != nil {
		return nil, err
	}

	//returns extensions
	return &ExtensionManager{
		S3Extension: s3Extension,
	}, nil
}
