// Copyright (c) 2022 by Reiform. All Rights Reserved.

package websockets

import (
	"context"
	"github.com/gorilla/websocket"
	"net/http"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"time"
)

//dataqueue entry
type queueEntry struct {
	uuid model.MynahUuid
	msg  []byte
}

//websocket server implements WebSocketProvider
type webSocketServer struct {
	//data to be distributed to clients
	dataChan chan queueEntry
	//lookup mapping connected client uuids to connections
	clients map[model.MynahUuid]*connectedClient
	//channel to accept new clients
	registerClientChan chan *connectedClient
	//channel for removing clients from lookup
	deregisterClientChan chan *connectedClient
	//server completion context
	ctx context.Context
	//server completion function
	cancel context.CancelFunc
}

//an authenticated client connected
type connectedClient struct {
	//the websocket connection
	conn *websocket.Conn
	//channel for outgoing messages
	outgoing chan []byte
	//the user id for this client (authenticated)
	uuid model.MynahUuid
	//the connected server
	connManager *webSocketServer
}

// NewWebSocketProvider create a new websocket provider
func NewWebSocketProvider(mynahSettings *settings.MynahSettings) WebSocketProvider {
	ctx, cancel := context.WithCancel(context.Background())
	return &webSocketServer{
		dataChan:             make(chan queueEntry, 256),
		clients:              make(map[model.MynahUuid]*connectedClient),
		registerClientChan:   make(chan *connectedClient),
		deregisterClientChan: make(chan *connectedClient),
		ctx:                  ctx,
		cancel:               cancel,
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
		if err := c.conn.Close(); err != nil {
			log.Warnf("error closing websocket client connection: %s", err)
		}
	}()

	//set the pong reader
	if err := c.conn.SetReadDeadline(time.Now().Add(60 * time.Second)); err != nil {
		log.Warnf("error setting websocket connection read deadline to 60 seconds: %s", err)
	}
	c.conn.SetPongHandler(func(string) error {
		return c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	})

	//write messages to the connected client
	for {
		select {
		case <-ticker.C:
			if err := c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second)); err != nil {
				log.Warnf("error setting websocket connection write deadline to 10 seconds: %s", err)
			}

			//send a ping message to the client
			if pingErr := c.conn.WriteMessage(websocket.PingMessage, nil); pingErr != nil {
				//exit goroutine and disconnect/deregister client
				log.Errorf("error sending ping to client: %s", pingErr)
				return
			}

		case msg := <-c.outgoing:
			if err := c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second)); err != nil {
				log.Warnf("error setting websocket connection write deadline to 10 seconds: %s", err)
			}

			//get the next websocket writer
			if writer, writerErr := c.conn.NextWriter(websocket.TextMessage); writerErr == nil {
				if _, err := writer.Write(msg); err != nil {
					log.Errorf("failed to write to websocket client: %s", err)
				}
				err := writer.Close()
				if err != nil {
					log.Warnf("error closing websocket connection writer: %s", err)
				}

			} else {
				log.Warnf("error sending message to websocket client: %s", writerErr)
				//exit goroutine and disconnect/deregister client
				return
			}
		}
	}
}

// ServerHandler return a handler used in a rest endpoint
func (w *webSocketServer) ServerHandler() middleware.HandlerFunc {
	//listen for new client data to distribute
	go func() {
		for {
			select {
			case <-w.ctx.Done():
				return
			case newClient := <-w.registerClientChan:
				log.Infof("registered websocket client %s", newClient.uuid)
				//register the client
				w.clients[newClient.uuid] = newClient

			case existingClient := <-w.deregisterClientChan:
				//remove the client
				if _, ok := w.clients[existingClient.uuid]; ok {
					delete(w.clients, existingClient.uuid)
					close(existingClient.outgoing)
				}
				log.Infof("deregistered websocket client %s", existingClient.uuid)

			case newMsg := <-w.dataChan:
				//find the client in the lookup
				if client, found := w.clients[newMsg.uuid]; found {
					//offload the writing to the client's goroutine
					client.outgoing <- newMsg.msg
				} else {
					log.Warnf("user %s is not connected as a websocket client", newMsg.uuid)
				}
			}
		}
	}()

	//standard websocket upgrader
	upgrader := websocket.Upgrader{}

	//return the http handler
	return func(ctx *middleware.Context) {
		//upgrade the http client request
		clientConn, err := upgrader.Upgrade(ctx.Writer, ctx.Request, nil)

		if err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to upgrade http client: %s", err)
			return
		}

		//create a client
		client := connectedClient{
			conn:        clientConn,
			outgoing:    make(chan []byte, 256),
			uuid:        ctx.User.Uuid,
			connManager: w,
		}

		//register the client
		w.registerClientChan <- &client

		//start the client writer goroutine (ensures synchronized writes)
		go client.clientWrite()
	}
}

// Send accept data to send to a connected client
func (w *webSocketServer) Send(uuid model.MynahUuid, msg []byte) {
	if (msg != nil) || (len(msg) == 0) {
		w.dataChan <- queueEntry{
			uuid: uuid,
			msg:  msg,
		}
	} else {
		log.Warnf("ignoring empty websocket message to %s", uuid)
	}
}

// Close close connected clients
func (w *webSocketServer) Close() {
	log.Infof("closing websocket connections")
	w.cancel()
	for uuid, client := range w.clients {
		close(client.outgoing)
		delete(w.clients, uuid)
	}
}
