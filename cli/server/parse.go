// Copyright (c) 2022 by Reiform. All Rights Reserved.

package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// RequestParseJson parse the contents of a request into a structure
func RequestParseJson(response *http.Response, target interface{}) error {
	contentType := response.Header.Get("Content-Type")
	//verify the content type
	if contentType != "application/json" {
		return fmt.Errorf("invalid response content type: %s", contentType)
	}

	//check for an empty/nil body
	if (response.Body == nil) || (response.Body == http.NoBody) {
		return errors.New("no body to decode")
	}

	//create a json decoder
	decoder := json.NewDecoder(response.Body)
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
			return errors.New("request body must not be larger than 1MB")

		default:
			return err
		}
	}

	//check for additional data
	if decoder.Decode(&struct{}{}) != io.EOF {
		return errors.New("request body must only contain a single JSON object")
	}
	return nil
}

// RequestSerializeJson write a struct to a request
func RequestSerializeJson(val interface{}) (io.Reader, error) {
	//marshal the response
	if jsonData, jsonErr := json.Marshal(val); jsonErr == nil {
		//respond with json
		return bytes.NewBuffer(jsonData), nil
	} else {
		return nil, fmt.Errorf("failed to generate json response %s", jsonErr)
	}
}
