// Copyright (c) 2022 by Reiform. All Rights Reserved.

package extensions

import "reiform.com/mynah/aws"

// ExtensionManager loads and provides access to extensions
type ExtensionManager struct {
	// S3Extension provides storage to aws s3
	S3Extension aws.S3Extension
}
