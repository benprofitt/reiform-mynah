package websockets

import (
	"github.com/gorilla/websocket"
	"net/http"
)

//Manages websocket, accepts data to send to clients
type WebSocketProvider interface {
	//create a handler that upgrades an http endpoint
	ServerHandler() http.HandlerFunc
	//accept data to send to a connected client
	Send(uuid *string, msg []byte)
}

//dataqueue entry
type queueEntry struct {
	uuid string
	msg  []byte
}

//an authenticated client connected
type connectedClient struct {
	//the websocket connection
	conn *websocket.Conn
	//channel for outgoing messages
	outgoing chan []byte
}

//websocket server adheres to WebSocketProvider
type webSocketServer struct {
	//data to be distributed to clients
	dataChan chan queueEntry
	//lookup mapping connected client uuids to connections
	clients map[string]connectedClient
}
