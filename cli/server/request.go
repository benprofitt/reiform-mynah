// Copyright (c) 2022 by Reiform. All Rights Reserved.

package server

import (
	"fmt"
	"io"
	"net/http"
	"path"
)

// NewRequest creates a new request for the mynah server
func (s MynahClient) NewRequest(method, url string, body io.Reader) (*http.Request, error) {
	fullUrl := fmt.Sprintf("%s://%s", s.proto, path.Join(s.serverAddress, s.urlPrefix, url))

	//create the request
	if req, err := http.NewRequest(method, fullUrl, body); err == nil {
		//add authentication
		req.Header.Set(s.jwtHeaderName, s.jwt)

		return req, nil
	} else {
		return nil, err
	}
}

// MakeRequest makes the request
func (s MynahClient) MakeRequest(req *http.Request) (*http.Response, error) {
	return s.client.Do(req)
}
