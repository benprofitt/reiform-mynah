// Copyright (c) 2022 by Reiform. All Rights Reserved.

package pyimpl

import "reiform.com/mynah/model"

// Apply the file metadata changes
func (d ImageMetadataResponse) apply(file *model.MynahFile, version *model.MynahFileVersion) error {
	version.Metadata[model.MetadataWidth] = d.Width
	version.Metadata[model.MetadataHeight] = d.Height
	version.Metadata[model.MetadataChannels] = d.Channels
	file.InitialMean = d.Mean
	file.InitialStdDev = d.StdDev
	return nil
}

// NewImageMetadataRequest creates a new image metadata request
func (p *localImplProvider) NewImageMetadataRequest(path string) *ImageMetadataRequest {
	return &ImageMetadataRequest{
		Path: path,
	}
}
