// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

//ipc provider
type IPCProvider interface {
	//handle new events
	HandleEvents(func(userUuid *string, msg []byte))
	//Close the ipc provider
	Close()
}
