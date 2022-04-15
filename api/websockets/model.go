// Copyright (c) 2022 by Reiform. All Rights Reserved.

package websockets

import (
	"net/http"
	"reiform.com/mynah/model"
)

// WebSocketProvider Manages websocket, accepts data to send to clients
type WebSocketProvider interface {
	// ServerHandler create a handler that upgrades an http endpoint
	ServerHandler() http.HandlerFunc
	// Send accept data to send to a connected client
	Send(uuid model.MynahUuid, msg []byte)
	// Close close connections
	Close()
}
