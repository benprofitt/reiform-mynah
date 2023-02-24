// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
	"reiform.com/mynah-api/models/dataset"
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

	newDataset := &dataset.MynahDataset{
		DatasetId:    types.NewMynahUuid(),
		DatasetName:  body.DatasetName,
		DateCreated:  time.Now().Unix(),
		DateModified: time.Now().Unix(),
		DatasetType:  dataset.MynahDatasetType(body.DatasetType),
		CreatedBy:    "no-user", // TODO no user ownership currently
	}

	if err := dataset.CreateMynahDataset(db.NewContext(), newDataset); err != nil {
		log.Info("DatasetCreate failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	ctx.JSON(http.StatusOK, newDataset)
}

// DatasetGet gets a dataset by id
func DatasetGet(ctx *gin.Context) {
	datasetId := ctx.Param("dataset_id")
	mynahDataset, found, err := dataset.GetMynahDataset(db.NewContext(), types.MynahUuid(datasetId))
	if err != nil {
		log.Info("DatasetGet failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}
	if !found {
		log.Info("DatasetGet failed, no such dataset: %s", datasetId)
		ctx.Status(http.StatusNotFound)
		return
	}

	ctx.JSON(http.StatusOK, mynahDataset)
}

// DatasetList lists datasets
func DatasetList(ctx *gin.Context) {
	opts, err := db.GetPaginationOptions(ctx)
	if err != nil {
		log.Info("DatasetList failed: %s", err)
		ctx.Status(http.StatusBadRequest)
		return
	}

	datasets, err := dataset.ListMynahDatasets(db.NewContext(), opts)
	if err != nil {
		log.Info("DatasetList failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	ctx.JSON(http.StatusOK, datasets)
}
