package async

import (
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/websockets"
	"testing"
	"time"
)

//run tests on the async architecture
func TestAsync(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//init the websocket server
	wsProvider := websockets.NewWebSocketProvider(mynahSettings)
	defer wsProvider.Close()

	//init the async engine
	asyncProvider := NewAsyncProvider(mynahSettings, wsProvider)
	defer asyncProvider.Close()

	taskCount := 10

	user := model.MynahUser{
		Uuid: "test_async_user",
	}

	resultChan := make(chan int, taskCount)

	//add some tasks
	for i := 0; i < taskCount; i++ {
		//start tasks
		asyncProvider.StartAsyncTask(&user, func(u *model.MynahUser) ([]byte, error) {
			time.Sleep(time.Second * 1)
			resultChan <- 0
			return nil, nil
		})
	}

	//wait for tasks to complete
	for i := 0; i < taskCount; i++ {
		<-resultChan
	}
}
