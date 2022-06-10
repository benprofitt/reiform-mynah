// Copyright (c) 2022 by Reiform. All Rights Reserved.

package tools

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reiform.com/mynah/log"
)

// WriteReaderToZip takes an io reader and writes to a zip archive
func WriteReaderToZip(writer *zip.Writer, reader io.Reader, zipPath string) error {
	//write the file to the zip archive
	zipFile, err := writer.Create(filepath.Clean(zipPath))
	if err != nil {
		return fmt.Errorf("failed to create file %s in zip archive: %s", zipPath, err)
	}

	//copy the contents over
	if _, err := io.Copy(zipFile, reader); err != nil {
		return fmt.Errorf("failed to copy local contents into zip archive: %s", err)
	}

	return nil
}

// WriteToZip writes a file by local path to a path in the given zip archive
func WriteToZip(writer *zip.Writer, localPath, zipPath string) error {
	localFile, err := os.Open(filepath.Clean(localPath))
	if err != nil {
		return fmt.Errorf("failed to open file: %s", err)
	}

	defer func(localFile *os.File) {
		err := localFile.Close()
		if err != nil {
			log.Warnf("failed to close file after writing to zip archive: %s", err)
		}
	}(localFile)

	return WriteReaderToZip(writer, localFile, zipPath)
}
