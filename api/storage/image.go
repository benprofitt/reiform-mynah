// Copyright (c) 2022 by Reiform. All Rights Reserved.

package storage

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"reiform.com/mynah/log"
)

//holds information about an image
type imageStat struct {
	//dimensions
	width  int
	height int

	//the format (as decoded)
	format string
}

//Accepted mime types
type MynahImageMimeType string

const (
	PNGType   MynahImageMimeType = "image/png"
	JPEGType  MynahImageMimeType = "image/jpeg"
	OtherType MynahImageMimeType = "other"
)

//attempt to infer the type
func PredictMimeType(detected string) MynahImageMimeType {
	if detected == "image/png" {
		return PNGType
	} else if detected == "image/jpeg" {
		return JPEGType
	} else {
		return OtherType
	}
}

//attempt to get the dimensions of a file as an image
func GetImageMetadata(localPath string, mimeType MynahImageMimeType) (*imageStat, error) {
	if (mimeType != PNGType) && (mimeType != JPEGType) {
		return nil, fmt.Errorf("unsupported image mime type: %s", mimeType)
	}

	//read in the image
	if file, err := os.Open(filepath.Clean(localPath)); err == nil {
		defer func() {
			if err := file.Close(); err != nil {
				log.Errorf("error closing file %s: %s", localPath, err)
			}
		}()

		var stat imageStat

		//attempt to decode
		im, format, err := image.DecodeConfig(file)
		if err != nil {
			return nil, err
		}
		stat.width = im.Width
		stat.height = im.Height
		stat.format = format

		return &stat, nil

	} else {
		return nil, err
	}
}
