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
	resp, err := s.client.Do(req)

	if (err == nil) && (resp.StatusCode != http.StatusOK) {
		return nil, fmt.Errorf("mynah server request failed with status: %s", resp.Status)
	}

	return resp, err
}

// ExecutePostJsonRequest creates a post request that takes json and receives json
func (s MynahClient) ExecutePostJsonRequest(path string, requestBody interface{}, responseBody interface{}) error {
	jsonData, err := RequestSerializeJson(requestBody)
	if err != nil {
		return fmt.Errorf("failed to create mynah server request: %s", err)
	}

	//create a new post request
	request, err := s.NewRequest("POST", path, jsonData)
	if err != nil {
		return fmt.Errorf("failed to create mynah server request: %s", err)
	}

	//set the content type
	request.Header.Set("Content-Type", "application/json")

	response, err := s.MakeRequest(request)
	if err != nil {
		return fmt.Errorf("failed to create mynah server request: %s", err)
	}

	//parse the response
	if err = RequestParseJson(response, responseBody); err != nil {
		return fmt.Errorf("failed to parse mynah server response: %s", err)
	}

	return nil
}
