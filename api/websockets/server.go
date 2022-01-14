package websockets

import (
	"github.com/gorilla/websocket"
	"net/http"
)

//return a handler used in a rest endpoint
func (w *WebSocketProvider) ServerHandler() http.HandlerFunc {
	//standard websocket upgrader
	upgrader := websocket.Upgrader{}

	//return the http handler
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		client, upgradeErr := upgrader.Upgrade(w, r, nil)
		if upgradeErr != nil {
			log.Print("failed to upgrade http client: ", upgradeErr)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		defer client.Close()

		//TODO broadcast messages to this client
	})
}
