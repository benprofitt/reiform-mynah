// Copyright (c) 2022 by Reiform. All Rights Reserved.

package json

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Decode the provided json data into the struct
func Decode(rc io.ReadCloser, target interface{}) error {
	//create a json decoder
	decoder := json.NewDecoder(rc)
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
