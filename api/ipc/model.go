// Copyright (c) 2022 by Reiform. All Rights Reserved.

package ipc

import "reiform.com/mynah/model"

// IPCServer ipc server
type IPCServer interface {
	// Listen for an ipc message
	Listen() ([]byte, error)
	// ListenMany accepts messages until closed
	ListenMany(func(userUuid model.MynahUuid, msg []byte))
	//Close the ipc provider
	Close()
}
