// Copyright (c) 2022 by Reiform. All Rights Reserved.

package websockets

import (
	"github.com/gorilla/websocket"
	"log"
	"net/http"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/settings"
	"time"
)

//dataqueue entry
type queueEntry struct {
	uuid string
	msg  []byte
}

//websocket server implements WebSocketProvider
type webSocketServer struct {
	//data to be distributed to clients
	dataChan chan queueEntry
	//lookup mapping connected client uuids to connections
	clients map[string]*connectedClient
	//channel to accept new clients
	registerClientChan chan *connectedClient
	//channel for removing clients from lookup
	deregisterClientChan chan *connectedClient
}

//an authenticated client connected
type connectedClient struct {
	//the websocket connection
	conn *websocket.Conn
	//channel for outgoing messages
	outgoing chan []byte
	//the user id for this client (authenticated)
	uuid string
	//the connected server
	connManager *webSocketServer
}

//create a new websocket provider
func NewWebSocketProvider(mynahSettings *settings.MynahSettings) WebSocketProvider {
	return &webSocketServer{
		dataChan:             make(chan queueEntry, 256),
		clients:              make(map[string]*connectedClient),
		registerClientChan:   make(chan *connectedClient),
		deregisterClientChan: make(chan *connectedClient),
	}
}

//write to a client connection
func (c *connectedClient) clientWrite() {
	//when to send ping messages (Note: must be more frequent than the pong deadline)
	ticker := time.NewTicker(45 * time.Second)
	defer func() {
		//stop the ping timer
		ticker.Stop()
		//deregister the client
		c.connManager.deregisterClientChan <- c
		//close the client connection
		c.conn.Close()
	}()

	//set the pong reader
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	//write messages to the connected client
	for {
		select {
		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

			//send a ping message to the client
			if pingErr := c.conn.WriteMessage(websocket.PingMessage, nil); pingErr != nil {
				//exit goroutine and disconnect/deregister client
				log.Printf("error sending ping to client: %s", pingErr)
				return
			}

		case msg := <-c.outgoing:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

			//get the next websocket writer
			if writer, writerErr := c.conn.NextWriter(websocket.TextMessage); writerErr == nil {
				if _, err := writer.Write(msg); err != nil {
					log.Printf("failed to write to websocket client: %s", err)
				}
				writer.Close()

			} else {
				log.Printf("error sending message to websocket client: %s", writerErr)
				//exit goroutine and disconnect/deregister client
				return
			}
		}
	}
}

//return a handler used in a rest endpoint
func (w *webSocketServer) ServerHandler() http.HandlerFunc {
	//listen for new client data to distribute
	go func() {
		for {
			select {
			case newClient := <-w.registerClientChan:
				log.Printf("registered websocket client %s", newClient.uuid)
				//register the client
				w.clients[newClient.uuid] = newClient

			case existingClient := <-w.deregisterClientChan:
				//remove the client
				if _, ok := w.clients[existingClient.uuid]; ok {
					delete(w.clients, existingClient.uuid)
					close(existingClient.outgoing)
				}
				log.Printf("deregistered websocket client %s", existingClient.uuid)

			case newMsg := <-w.dataChan:
				//find the client in the lookup
				if client, found := w.clients[newMsg.uuid]; found {
					//offload the writing to the client's goroutine
					client.outgoing <- newMsg.msg
				} else {
					log.Printf("user %s is not connected as a websocket client", newMsg.uuid)
				}
			}
		}
	}()

	//standard websocket upgrader
	upgrader := websocket.Upgrader{}

	//return the http handler
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//upgrade the http client request
		clientConn, upgradeErr := upgrader.Upgrade(writer, request, nil)

		if upgradeErr != nil {
			log.Print("failed to upgrade http client: ", upgradeErr)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//the authenticated user
		user := middleware.GetUserFromRequest(request)

		//create a client
		client := connectedClient{
			conn:        clientConn,
			outgoing:    make(chan []byte, 256),
			uuid:        user.Uuid,
			connManager: w,
		}

		//register the client
		w.registerClientChan <- &client

		//start the client writer goroutine (ensures synchronized writes)
		go client.clientWrite()
	})
}

//accept data to send to a connected client
func (w *webSocketServer) Send(uuid *string, msg []byte) {
	w.dataChan <- queueEntry{
		uuid: *uuid,
		msg:  msg,
	}
}

//close connected clients
func (w *webSocketServer) Close() {
	log.Printf("closing websocket connections")
	for uuid, client := range w.clients {
		close(client.outgoing)
		delete(w.clients, uuid)
	}
}
