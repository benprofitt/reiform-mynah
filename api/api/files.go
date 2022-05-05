// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"fmt"
	"github.com/gabriel-vasile/mimetype"
	"github.com/gorilla/mux"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/pyimpl"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"time"
)

const fileKey string = "file"
const fileVersionIdKey string = "version_id"
const MultipartFormFileKey = "file"

//check if a file is of a valid type
func validFiletype(filetype *string) bool {
	//TODO make this more granular
	return true
}

//check if this is an image mime type
func isImageType(mType *mimetype.MIME) bool {
	switch mType.String() {
	case "image/png":
		return true
	case "image/jpeg":
		return true
	case "image/tiff":
		return true
	default:
		return false
	}
}

//accept a file upload, save the file using the storage provider, and reference as part of the dataset
func handleFileUpload(mynahSettings *settings.MynahSettings,
	dbProvider db.DBProvider,
	storageProvider storage.StorageProvider,
	pyImplProvider pyimpl.PyImplProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the user uploading the file
		user := middleware.GetUserFromRequest(request)

		//requires a post request
		if request.Method != "POST" {
			//bad request
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//attempt to parse the multipart form
		if err := request.ParseMultipartForm(mynahSettings.StorageSettings.MaxUpload); err != nil {
			log.Warnf("failed to parse multipart form: %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//get the file
		file, fileHeader, formErr := request.FormFile(MultipartFormFileKey)
		if formErr != nil {
			log.Warnf("failed to get file from form: %s", formErr)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		defer func() {
			if err := file.Close(); err != nil {
				log.Warnf("error closing file from upload form: %s", err)
			}
		}()

		//check that the file is not too big
		if fileHeader.Size > mynahSettings.StorageSettings.MaxUpload {
			log.Warnf("file surpasses max upload size, ignoring")
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//check that the file can be read successfully
		fileContents, readErr := ioutil.ReadAll(file)
		if readErr != nil {
			log.Warnf("invalid file when reading: %s", readErr)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		detectedType := http.DetectContentType(fileContents)
		//validate the content type
		if !validFiletype(&detectedType) {
			log.Infof("file has unsupported content type %s, ignoring", detectedType)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		mynahFile, err := dbProvider.CreateFile(user, func(f *model.MynahFile) error {
			f.Name = "<unknown>" //TODO read this from the multipart form
			//write the contents of the file to storage
			if err := storageProvider.StoreFile(f, func(osFile *os.File) error {
				_, err := osFile.Write(fileContents)
				return err
			}); err != nil {
				return err
			}

			//determine the dimensions, channels of the file (if applicable)
			return storageProvider.GetStoredFile(f, model.OriginalVersionId, func(path string) error {
				//check the mime type for image metadata
				if mimeType, err := mimetype.DetectFile(filepath.Clean(path)); err == nil {
					if isImageType(mimeType) {
						if err = pyImplProvider.ImageMetadata(user, path, f, f.Versions[model.OriginalVersionId]); err != nil {
							log.Warnf("failed to get image metadata for %s: %s", f.Uuid, err)
						}
					} else {
						log.Infof("skipping metadata read for non image file type: %s", mimeType.String())
					}

				} else {
					log.Warnf("failed to read mime type for file %s: %s", f.Uuid, err)
				}

				return nil
			})
		})

		//create the file in the database
		if err != nil {
			log.Errorf("failed to add file to database %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//return the file metadata
		if err := responseWriteJson(writer, &mynahFile); err != nil {
			log.Errorf("failed to generate json response for file upload %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

//handler that loads a file and serves the contents
func handleViewFile(dbProvider db.DBProvider, storageProvider storage.StorageProvider) http.HandlerFunc {
	return func(writer http.ResponseWriter, request *http.Request) {
		//get the user from context
		user := middleware.GetUserFromRequest(request)

		//get the file id
		if fileId, ok := mux.Vars(request)[fileKey]; ok {
			fileVersionId := model.LatestVersionId
			//check for the file version id
			if versionIdStr, ok := mux.Vars(request)[fileVersionIdKey]; ok {
				fileVersionId = model.MynahFileVersionId(versionIdStr)
			}

			//load the file metadata
			if file, fileErr := dbProvider.GetFile(model.MynahUuid(fileId), user); fileErr == nil {
				//serve the file contents
				storeErr := storageProvider.GetStoredFile(file, fileVersionId, func(path string) error {
					//open the file
					osFile, osErr := os.Open(filepath.Clean(path))
					if osErr != nil {
						return fmt.Errorf("failed to open file %s: %s", file.Uuid, osErr)
					}

					defer func() {
						if err := osFile.Close(); err != nil {
							log.Errorf("error closing file %s: %s", file.Uuid, err)
						}
					}()

					modTime := time.Unix(file.DateCreated, 0)

					//determine the last modified time
					http.ServeContent(writer, request, file.Name, modTime, osFile)
					return nil
				})

				if storeErr != nil {
					log.Errorf("error writing file to response: %s", storeErr)
					writer.WriteHeader(http.StatusInternalServerError)
				}

			} else {
				log.Warnf("error retrieving file %s: %s", fileId, fileErr)
				writer.WriteHeader(http.StatusBadRequest)
			}

		} else {
			log.Errorf("file request path missing %s", fileKey)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	}
}
