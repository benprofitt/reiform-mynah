package websockets

import (
	//"github.com/gorilla/websocket"
	"net/http"
	"reiform.com/mynah/settings"
)

//create a new websocket provider
func NewWebSocketProvider(mynahSettings *settings.MynahSettings) WebSocketProvider {
	return &webSocketServer{

	}
}

//return a handler used in a rest endpoint
func (w *webSocketServer) ServerHandler() http.HandlerFunc {

	//TODO create a goroutine that listens for new data to send to clients

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
func (w *webSocketServer) Send(uuid *string, msg []byte) error {
	return nil
}
