// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"fmt"
	"github.com/gabriel-vasile/mimetype"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"time"
)

const FileKey string = "file"
const FileVersionIdKey string = "version_id"
const MultipartFormFileKey = "file"
const MultipartFormNameKey = "name"
const FileIdKey string = "fileid"

//check if a file is of a valid type
func validFiletype(filetype *string) bool {
	//TODO make this more granular
	return true
}

// handleFileUpload accepts a file upload, save the file using the storage provider, and reference as part of the dataset
func handleFileUpload(mynahSettings *settings.MynahSettings,
	dbProvider db.DBProvider,
	storageProvider storage.StorageProvider) middleware.HandlerFunc {
	return func(ctx *middleware.Context) {
		//requires a post request
		if ctx.Request.Method != "POST" {
			//bad request
			ctx.Writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//attempt to parse the multipart form
		if err := ctx.Request.ParseMultipartForm(mynahSettings.StorageSettings.MaxUpload); err != nil {
			ctx.Error(http.StatusBadRequest, "failed to parse multipart form: %s", err)
			return
		}

		//get the file
		file, fileHeader, err := ctx.Request.FormFile(MultipartFormFileKey)
		if err != nil {
			ctx.Error(http.StatusBadRequest, "failed to get file from form: %s", err)
			return
		}
		defer func() {
			if err := file.Close(); err != nil {
				log.Warnf("error closing file from upload form: %s", err)
			}
		}()

		//check that the file is not too big
		if fileHeader.Size > mynahSettings.StorageSettings.MaxUpload {
			ctx.Error(http.StatusBadRequest, "file surpasses max upload size, ignoring")
			return
		}

		//check that the file can be read successfully
		fileContents, err := ioutil.ReadAll(file)
		if err != nil {
			ctx.Error(http.StatusBadRequest, "invalid file when reading: %s", err)
			return
		}

		detectedType := http.DetectContentType(fileContents)
		//validate the content type
		if !validFiletype(&detectedType) {
			ctx.Error(http.StatusBadRequest, "file has unsupported content type %s, ignoring", detectedType)
			return
		}

		mynahFile, err := dbProvider.CreateFile(ctx.User, func(f *model.MynahFile) error {
			f.Name = fileHeader.Filename
			//write the contents of the file to storage
			if err := storageProvider.StoreFile(f, func(osFile *os.File) error {
				_, err := osFile.Write(fileContents)
				return err
			}); err != nil {
				return err
			}

			//determine the mime type
			return storageProvider.GetStoredFile(f.Uuid, f.Versions[model.OriginalVersionId], func(localFile storage.MynahLocalFile) error {
				//check the mime type for image metadata
				if mimeType, err := mimetype.DetectFile(filepath.Clean(localFile.Path())); err == nil {
					f.UploadMimeType = mimeType.String()
				} else {
					log.Warnf("failed to read mime type for file %s: %s", f.Uuid, err)
				}

				return nil
			})
		})

		//create the file in the database
		if err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to add file to database %s", err)
			return
		}

		//return the file metadata
		if err := ctx.WriteJson(&mynahFile); err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to generate json response for file upload %s", err)
			return
		}
	}
}

// handleViewFile that loads a file and serves the contents
func handleViewFile(dbProvider db.DBProvider, storageProvider storage.StorageProvider) middleware.HandlerFunc {
	return func(ctx *middleware.Context) {
		//get the file id
		fileId := ctx.Vars()[FileKey]

		fileVersionId := model.LatestVersionId
		//check for the file version id
		if versionIdStr, ok := ctx.Vars()[FileVersionIdKey]; ok {
			fileVersionId = model.MynahFileVersionId(versionIdStr)
		}

		file, err := dbProvider.GetFile(model.MynahUuid(fileId), ctx.User)
		if err != nil {
			ctx.Error(http.StatusNotFound, "failed to get file: %s", err)
			return
		}

		// verify that this version of the file exists
		fileVersion, err := file.GetFileVersion(fileVersionId)
		if err != nil {
			ctx.Error(http.StatusNotFound, "no such file version: %s", err)
			return
		}

		//serve the file contents
		storeErr := storageProvider.GetStoredFile(file.Uuid, fileVersion, func(localFile storage.MynahLocalFile) error {
			//open the file
			osFile, osErr := os.Open(filepath.Clean(localFile.Path()))
			if osErr != nil {
				return fmt.Errorf("failed to open file %s: %s", file.Uuid, osErr)
			}

			defer func() {
				if err := osFile.Close(); err != nil {
					log.Errorf("error closing file %s: %s", file.Uuid, err)
				}
			}()

			//determine the last modified time
			ctx.ServeContent(file.Name, time.Unix(file.DateCreated, 0), osFile)
			return nil
		})

		if storeErr != nil {
			ctx.Error(http.StatusInternalServerError, "error serving file: %s", err)
			return
		}
	}
}

// handleListFileMetadata  gets metadata for a list of files
func handleListFileMetadata(dbProvider db.DBProvider) middleware.HandlerFunc {
	return func(ctx *middleware.Context) {
		if err := ctx.Request.ParseForm(); err != nil {
			ctx.Error(http.StatusBadRequest, "failed to parse http form for list files request")
			return
		}

		ids, ok := ctx.Request.Form[FileIdKey]

		if !ok {
			ctx.Error(http.StatusBadRequest, "list file metadata missing ids given with key '%s'", FileIdKey)
			return
		}

		fileIds := make([]model.MynahUuid, len(ids))
		for i := 0; i < len(ids); i++ {
			fileIds[i] = model.MynahUuid(ids[i])
		}

		fileRes, err := dbProvider.GetFiles(fileIds, ctx.User)
		if err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to request files: %s", err)
			return
		}

		//write the response
		if err := ctx.WriteJson(&fileRes); err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to write response as json: %s", err)
			return
		}
	}
}
