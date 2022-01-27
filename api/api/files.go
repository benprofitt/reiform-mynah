// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"encoding/json"
	"github.com/google/uuid"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"reiform.com/mynah/db"
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
			log.Printf("failed to parse multipart form: %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//get the file
		file, fileHeader, formErr := request.FormFile("file")
		if formErr != nil {
			log.Printf("failed to get file from form: %s", formErr)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		defer file.Close()

		//check that the file is not too big
		if fileHeader.Size > mynahSettings.StorageSettings.MaxUpload {
			log.Printf("file surpasses max upload size, ignoring")
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//check that the file can be read successfully
		fileContents, readErr := ioutil.ReadAll(file)
		if readErr != nil {
			log.Printf("invalid file when reading: %s", readErr)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		detectedType := http.DetectContentType(fileContents)
		//validate the content type
		if !validFiletype(&detectedType) {
			log.Printf("file has unsupported content type %s, ignoring", detectedType)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//create a new file db entry
		mynahFile := model.MynahFile{
			Uuid: uuid.NewString(),
		}

		//write the contents of the file to storage
		storeErr := storageProvider.StoreFile(&mynahFile, func(f *os.File) error {
			_, err := f.Write(fileContents)
			return err
		})

		if storeErr != nil {
			log.Printf("failed to write file to local storage %s", storeErr)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//add the file to the database
		if err := dbProvider.CreateFile(&mynahFile, user); err != nil {
			log.Printf("failed to add file to database %s", err)
			//remove the file from local storage
			if dErr := storageProvider.DeleteFile(&mynahFile); dErr != nil {
				log.Printf("failed to delete file from local storage %s", dErr)
			}
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//return the file metadata
		if jsonResp, jsonErr := json.Marshal(&mynahFile); jsonErr == nil {
			if _, writeErr := writer.Write(jsonResp); writeErr != nil {
				log.Printf("failed to write json as response for file upload %s", writeErr)
				writer.WriteHeader(http.StatusInternalServerError)
			}
			//set content type to json
			writer.Header().Set("Content-Type", "application/json")

		} else {
			log.Printf("failed to generate json response for file upload %s", jsonErr)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
