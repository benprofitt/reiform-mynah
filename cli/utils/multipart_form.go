// Copyright (c) 2022 by Reiform. All Rights Reserved.

package utils

import (
	"bytes"
	"io"
	"mime/multipart"
	"os"
	"path/filepath"
	"reiform.com/mynah/api"
	"reiform.com/mynah/log"
)

// CreateMultipartForm creates a multipart upload from a reader
func CreateMultipartForm(localPath string) (bytes.Buffer, string, error) {
	var buff bytes.Buffer

	//try to open the file
	file, err := os.Open(filepath.Clean(localPath))
	if err != nil {
		return buff, "", err
	}

	defer func(file *os.File, localPath string) {
		err := file.Close()
		if err != nil {
			log.Warnf("failed to close file: %s", localPath)
		}
	}(file, localPath)

	//create a multipart writer
	writer := multipart.NewWriter(&buff)

	fw, err := writer.CreateFormFile(api.MultipartFormFileKey, file.Name())
	if err != nil {
		return buff, "", err
	}

	//copy the file contents
	if _, err = io.Copy(fw, file); err != nil {
		return buff, "", err
	}

	//return the buffer, content type, and error if the writer can't be closed
	return buff, writer.FormDataContentType(), writer.Close()
}
