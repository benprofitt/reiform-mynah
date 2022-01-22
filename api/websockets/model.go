package websockets

import (
	"net/http"
)

//Manages websocket, accepts data to send to clients
type WebSocketProvider interface {
	//create a handler that upgrades an http endpoint
	ServerHandler() http.HandlerFunc
	//accept data to send to a connected client
	Send(uuid *string, msg []byte)
}
