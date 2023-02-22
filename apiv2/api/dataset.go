// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
	"reiform.com/mynah-api/models"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/services/log"
	"reiform.com/mynah-api/types"
	"time"
)

// DatasetCreateBody defines the request body for DatasetCreate
type DatasetCreateBody struct {
	DatasetName string `json:"dataset_name" binding:"required"`
	DatasetType string `json:"dataset_type" binding:"required;In(image_classification)"`
}

// DatasetCreate creates a new dataset
func DatasetCreate(ctx *gin.Context) {
	var body DatasetCreateBody
	if err := ctx.ShouldBindJSON(&body); err != nil {
		log.Info("DatasetCreate failed: %s", err)
		ctx.Status(http.StatusBadRequest)
		return
	}

	newDataset := &models.MynahDataset{
		DatasetId:    types.NewMynahUuid(),
		DatasetName:  body.DatasetName,
		DateCreated:  time.Now().Unix(),
		DateModified: time.Now().Unix(),
		DatasetType:  models.MynahDatasetType(body.DatasetType),
	}

	if err := models.CreateMynahDataset(db.NewContext(), newDataset); err != nil {
		log.Info("DatasetCreate failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	ctx.JSON(http.StatusOK, newDataset)
}
