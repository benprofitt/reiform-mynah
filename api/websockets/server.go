package websockets

import (
	"github.com/gorilla/websocket"
	"log"
	"net/http"
	"reiform.com/mynah/settings"
	"time"
)

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

//create a new websocket provider
func NewWebSocketProvider(mynahSettings *settings.MynahSettings) WebSocketProvider {
	return &webSocketServer{
		dataChan: make(chan queueEntry),
		clients:  make(map[string]connectedClient),
	}
}

//return a handler used in a rest endpoint
func (w *webSocketServer) ServerHandler() http.HandlerFunc {
	//when to send ping messages
	ticker := time.NewTicker(45 * time.Second)

	//listen for new client data to distribute
	go func() {
		for {
			select {
			case newMsg := <-w.dataChan:
				//find the client in the lookup
				if client, found := w.clients[newMsg.uuid]; found {
					//allow ten seconds for writing to client
					client.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

					//get the next websocket writer
					writer, writerErr := client.conn.NextWriter(websocket.TextMessage)
					if writerErr != nil {
						log.Printf("error sending message to websocket client %s", writerErr)

					} else {
						writer.Write(newMsg.msg)
						writer.Close()
					}

				} else {
					log.Printf("user %s is not connected as a websocket client", newMsg.uuid)
				}

			case <-ticker.C:
				//send a ping to each client
				for uuid, client := range w.clients {
					//allow ten seconds for writing to client
					client.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

					//send a ping message
					if pingErr := client.conn.WriteMessage(websocket.PingMessage, nil); pingErr != nil {
						log.Printf("error sending ping to client %s", uuid)
					}
				}
			}
		}
	}()

	//standard websocket upgrader
	//upgrader := websocket.Upgrader{}

	//return the http handler
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		// client, upgradeErr := upgrader.Upgrade(w, r, nil)
		// if upgradeErr != nil {
		// 	log.Print("failed to upgrade http client: ", upgradeErr)
		// 	writer.WriteHeader(http.StatusInternalServerError)
		// 	return
		// }
		// defer client.Close()
		//
		// //TODO broadcast messages to this client
	})
}

//accept data to send to a connected client
func (w *webSocketServer) Send(uuid *string, msg []byte) {
	w.dataChan <- queueEntry{
		uuid: *uuid,
		msg:  msg,
	}
}
