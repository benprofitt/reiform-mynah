// Copyright (c) 2022 by Reiform. All Rights Reserved.

package server

import (
	"github.com/gorilla/websocket"
	"net/http"
	"net/url"
	"path"
	"reiform.com/mynah/log"
)

// WsRes represents a websocket result
type WsRes struct {
	Error  error
	Result []byte
}

// WebsocketListener listens for websocket message(s), calls function on each
func (s MynahClient) WebsocketListener(expectMessages int, resChan chan WsRes) {
	//make a websocket request
	u := url.URL{
		Scheme: "ws",
		Host:   s.serverAddress,
		Path:   path.Join(s.urlPrefix, "websocket"),
	}

	//add the auth header
	headers := make(http.Header)
	headers.Add(s.jwtHeaderName, s.jwt)

	//connect to the server
	c, _, err := websocket.DefaultDialer.Dial(u.String(), headers)
	if err != nil {
		resChan <- WsRes{
			Error: err,
		}
		return
	}
	defer func(c *websocket.Conn) {
		err := c.Close()
		if err != nil {
			log.Warnf("failed to close websocket connection: %s", err)
		}
	}(c)

	//call the handler on the message
	for i := 0; i < expectMessages; i++ {
		if _, message, err := c.ReadMessage(); err == nil {
			resChan <- WsRes{
				Result: message,
			}
		} else {
			resChan <- WsRes{
				Error: err,
			}
			return
		}
	}
}
