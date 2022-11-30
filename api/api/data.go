// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

const idKey = "id"

// getDataJSON get data as json
func getDataJSON(dbProvider db.DBProvider) middleware.HandlerFunc {
	return func(ctx *middleware.Context) {

		dataId, ok := ctx.Vars()[idKey]
		//get request params
		if !ok {
			ctx.Error(http.StatusBadRequest, "data as json request path missing %s key", idKey)
			return
		}

		//request the binary data
		binData, err := dbProvider.GetBinObject(model.MynahUuid(dataId), ctx.User)
		if err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to get data as json object %s: %s", dataId, err)
			return
		}

		//respond with the json contents
		if err := ctx.WriteJsonBytes(binData.Data); err != nil {
			ctx.Error(http.StatusInternalServerError, "failed to write binary data", err)
		}
	}
}
