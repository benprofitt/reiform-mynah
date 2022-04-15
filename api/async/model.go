// Copyright (c) 2022 by Reiform. All Rights Reserved.

package async

import (
	"reiform.com/mynah/model"
)

// AsyncTaskHandler Handler to invoke asynchronously
type AsyncTaskHandler func(model.MynahUuid) ([]byte, error)

// AsyncProvider interface for launching new background processes
type AsyncProvider interface {
	// StartAsyncTask start processing an async task
	StartAsyncTask(*model.MynahUser, AsyncTaskHandler)
	// Close close the async task provider
	Close()
}
