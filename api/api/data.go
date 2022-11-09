// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gorilla/mux"
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

const idKey = "id"

// GetDataJSON get data as json
func GetDataJSON(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		dataId, ok := mux.Vars(request)[idKey]
		//get request params
		if !ok {
			log.Errorf("data as json request path missing %s key", idKey)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//request the binary data
		binData, err := dbProvider.GetBinObject(model.MynahUuid(dataId), user)
		if err != nil {
			log.Errorf("failed to get data as json object %s: %s", dataId, err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//respond with the json contents
		writer.Header().Add("Content-Type", "application/json")

		if _, err := writer.Write(binData.Data); err != nil {
			log.Errorf("failed to write report as json: %s", err)
		}
	})
}
