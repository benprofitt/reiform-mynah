// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"io/ioutil"
	"net/http"
	"os"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
)

//check if a file is of a valid type
func validFiletype(filetype *string) bool {
	//TODO make this more granular
	return true
}

//accept a file upload, save the file using the storage provider, and reference as part of the project
func handleFileUpload(mynahSettings *settings.MynahSettings, dbProvider db.DBProvider, storageProvider storage.StorageProvider) http.HandlerFunc {
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
		file, fileHeader, formErr := request.FormFile("file")
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

		mynahFile, err := dbProvider.CreateFile(user, func(f *model.MynahFile) {
			f.Name = "<unknown>" //TODO read this from the multipart form
		})

		//create the file in the database
		if err != nil {
			log.Errorf("failed to add file to database %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//write the contents of the file to storage
		storeErr := storageProvider.StoreFile(mynahFile, user, func(f *os.File) error {
			_, err := f.Write(fileContents)
			return err
		})

		if storeErr != nil {
			//try to delete file from database (ignore failure, already in a failure mode)
			_ = dbProvider.DeleteFile(&mynahFile.Uuid, user)

			log.Errorf("failed to write file to local storage %s", storeErr)
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
