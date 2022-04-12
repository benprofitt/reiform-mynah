// Copyright (c) 2022 by Reiform. All Rights Reserved.

package server

import "net/http"

// MynahClient defines the address of the Mynah server to communicate with
type MynahClient struct {
	//the http client
	client *http.Client
	//the protocol
	proto string
	//the address
	serverAddress string
	//the url prefix (i.e. /api/v1)
	urlPrefix string
	//the name of the header used to pass the jwt
	jwtHeaderName string
	//the jwt for the user making these requests
	jwt string
}

// NewMynahClient creates a new mynah server client
func NewMynahClient(address, jwt, proto string) *MynahClient {
	return &MynahClient{
		client:        &http.Client{},
		proto:         proto,
		serverAddress: address,
		urlPrefix:     "/api/v1",
		jwtHeaderName: "api-key",
		jwt:           jwt,
	}
}
