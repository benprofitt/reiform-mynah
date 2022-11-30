// Copyright (c) 2022 by Reiform. All Rights Reserved.

package middleware

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/gorilla/mux"
	"io"
	"net/http"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"strings"
	"time"
)

const maxBodySize1MB = 1048576

// Context defines the context of an http request
type Context struct {
	// User is the current authenticated user
	User *model.MynahUser

	// Writer is the http response writer
	Writer http.ResponseWriter
	// Request is the http request
	Request *http.Request
}

// NewContext creates a new context
func NewContext(writer http.ResponseWriter, request *http.Request) *Context {
	return &Context{
		Writer:  writer,
		Request: request,
	}
}

//verify that the content type of the request is json
func (ctx *Context) jsonContentType() bool {
	return ctx.Request.Header.Get("Content-Type") == "application/json"
}

// ReadJson from the requst
func (ctx *Context) ReadJson(target interface{}) error {
	//verify the content type
	if !ctx.jsonContentType() {
		return errors.New("invalid content type (not application/json)")
	}

	//check for an empty/nil body
	if (ctx.Request.Body == nil) || (ctx.Request.Body == http.NoBody) {
		return errors.New("no body to decode")
	}

	//max size is 1MB
	ctx.Request.Body = http.MaxBytesReader(ctx.Writer, ctx.Request.Body, maxBodySize1MB)

	//create a json decoder
	decoder := json.NewDecoder(ctx.Request.Body)
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

// WriteJson to the response
func (ctx *Context) WriteJson(body interface{}) error {
	//marshal the response
	if res, err := json.Marshal(body); err == nil {
		//respond with json
		ctx.Writer.Header().Add("Content-Type", "application/json")

		_, writeErr := ctx.Writer.Write(res)
		return writeErr
	} else {
		return fmt.Errorf("failed to generate json response %s", err)
	}
}

// Error responds with some error and logs the result
func (ctx *Context) Error(statusCode int, logFormat string, args ...interface{}) {
	log.Errorf(logFormat, args...)
	ctx.Writer.WriteHeader(statusCode)
}

// Vars gets the route variables for the request (if exist)
func (ctx *Context) Vars() map[string]string {
	return mux.Vars(ctx.Request)
}

// WriteJsonBytes writs binary data with a json content type
func (ctx *Context) WriteJsonBytes(data []byte) error {
	ctx.Writer.Header().Add("Content-Type", "application/json")
	_, err := ctx.Writer.Write(data)
	return err
}

// GetForm gets the first form value
func (ctx *Context) GetForm(key string) string {
	return ctx.Request.Form.Get(key)
}

// ServeContent replies to the request using the given read seeker
func (ctx *Context) ServeContent(name string, modTime time.Time, content io.ReadSeeker) {
	http.ServeContent(ctx.Writer, ctx.Request, name, modTime, content)
}
