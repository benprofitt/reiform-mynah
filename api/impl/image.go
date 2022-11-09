// Copyright (c) 2022 by Reiform. All Rights Reserved.

package impl

import (
	"fmt"
	"reiform.com/mynah/model"
	"reiform.com/mynah/storage"
)

// Apply the file metadata changes
func (d ImageMetadataResponse) apply(files map[model.MynahUuid]storage.MynahLocalFile) error {
	for fileId, data := range d.Images {
		// make sure the file exists, update
		if localFile, exists := files[fileId]; exists {
			version := localFile.FileVersion()
			version.Metadata[model.MetadataWidth] = data.Width
			version.Metadata[model.MetadataHeight] = data.Height
			version.Metadata[model.MetadataChannels] = data.Channels
			version.Metadata[model.MetadataMean] = data.Mean
			version.Metadata[model.MetadataStddev] = data.StdDev
		} else {
			return fmt.Errorf("get_metadata_for_images() returned data for an unknown file: %s", fileId)
		}
	}
	return nil
}

// NewBatchImageMetadataRequest creates a new image metadata request
func (p *localImplProvider) NewBatchImageMetadataRequest(files storage.MynahLocalFileSet) *ImageMetadataRequest {
	imageData := make([]ImageMetadataRequestLocalFile, 0, len(files))

	for id, file := range files {
		imageData = append(imageData, ImageMetadataRequestLocalFile{
			Uuid: id,
			Path: file.Path(),
		})
	}

	return &ImageMetadataRequest{
		Images: imageData,
	}
}
