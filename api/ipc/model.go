// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

import "reiform.com/mynah/model"

// IPCProvider ipc provider
type IPCProvider interface {
	// HandleEvents handle new events
	HandleEvents(func(userUuid model.MynahUuid, msg []byte))
	//Close the ipc provider
	Close()
}
