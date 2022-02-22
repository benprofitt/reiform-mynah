// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const maxBodySize1MB = 1048576

//verify that the content type of a request is json
func jsonContentType(request *http.Request) bool {
	return request.Header.Get("Content-Type") == "application/json"
}

//parse the contents of a request into a structure
func requestParseJson(writer http.ResponseWriter, request *http.Request, target interface{}) error {
	//verify the content type
	if !jsonContentType(request) {
		return errors.New("invalid content type (not application/json)")
	}

	//check for an empty/nil body
	if (request.Body == nil) || (request.Body == http.NoBody) {
		return errors.New("no body to decode")
	}

	//max size is 1MB
	request.Body = http.MaxBytesReader(writer, request.Body, maxBodySize1MB)

	//create a json decoder
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(target); err != nil {
		var syntaxError *json.SyntaxError
		var unmarshalTypeError *json.UnmarshalTypeError

		//errors as defined here: https://www.alexedwards.net/blog/how-to-properly-parse-a-json-request-body
		switch {
		case errors.As(err, &syntaxError):
			return fmt.Errorf("request body contains badly-formed JSON (at position %d)", syntaxError.Offset)

		case errors.Is(err, io.ErrUnexpectedEOF):
			return fmt.Errorf("request body contains badly-formed JSON")

		case errors.As(err, &unmarshalTypeError):
			return fmt.Errorf("request body contains an invalid value for the %q field (at position %d)", unmarshalTypeError.Field, unmarshalTypeError.Offset)

		case strings.HasPrefix(err.Error(), "json: unknown field "):
			fieldName := strings.TrimPrefix(err.Error(), "json: unknown field ")
			return fmt.Errorf("request body contains unknown field %s", fieldName)

		case errors.Is(err, io.EOF):
			return errors.New("request body must not be empty")

		case err.Error() == "http: request body too large":
			return errors.New("Request body must not be larger than 1MB")

		default:
			return err
		}
	}

	//check for additional data
	if decoder.Decode(&struct{}{}) != io.EOF {
		return errors.New("Request body must only contain a single JSON object")
	}
	return nil
}

//write a struct to a json response
func responseWriteJson(writer http.ResponseWriter, target interface{}) error {
	//marshal the response
	if jsonResp, jsonErr := json.Marshal(target); jsonErr == nil {
		//write response
		if _, writeErr := writer.Write(jsonResp); writeErr == nil {
			//respond with json
			writer.Header().Set("Content-Type", "application/json")
			return nil
		} else {
			return fmt.Errorf("failed to write response: %s", writeErr)
		}

	} else {
		return fmt.Errorf("failed to generate json response %s", jsonErr)
	}
}
