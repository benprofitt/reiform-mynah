package async

import (
	"reiform.com/mynah/model"
)

//Handler to invoke asynchronously
type AsyncTaskHandler func(*model.MynahUser) ([]byte, error)

//interface for launching new background processes
type AsyncProvider interface {
	//start processing an async task
	StartAsyncTask(*model.MynahUser, AsyncTaskHandler)
	//close the async task provider
	Close()
}
