// Copyright (c) 2022 by Reiform. All Rights Reserved.

package websockets

import (
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

// WebSocketProvider Manages websocket, accepts data to send to clients
type WebSocketProvider interface {
	// ServerHandler create a handler that upgrades an http endpoint
	ServerHandler() middleware.HandlerFunc
	// Send accept data to send to a connected client
	Send(uuid model.MynahUuid, msg []byte)
	// Close connections
	Close()
}
