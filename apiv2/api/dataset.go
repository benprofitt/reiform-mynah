// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
	"path/filepath"
	"reiform.com/mynah-api/models"
	"reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/models/types"
	"reiform.com/mynah-api/services/log"
	"reiform.com/mynah-api/settings"
	"time"
)

// DatasetCreateBody defines the request body for DatasetCreate
type DatasetCreateBody struct {
	DatasetName string `json:"dataset_name" binding:"required"`
	DatasetType string `json:"dataset_type" binding:"required;In(image_classification)"`
}

// DatasetCreate creates a new dataset
func DatasetCreate(ctx *gin.Context) {
	appCtx := getAppContext(ctx)

	var body DatasetCreateBody
	if err := ctx.ShouldBindJSON(&body); err != nil {
		log.Info("DatasetCreate failed: %s", err)
		ctx.Status(http.StatusBadRequest)
		return
	}

	newDataset := dataset.MynahDataset{
		DatasetId:    types.NewMynahUuid(),
		DatasetName:  body.DatasetName,
		DateCreated:  time.Now().Unix(),
		DateModified: time.Now().Unix(),
		DatasetType:  dataset.MynahDatasetType(body.DatasetType),
		CreatedBy:    "no-user", // TODO no user ownership currently
	}

	err := db.NewContext().NewTransaction(func(tx *db.Context) error {
		if err := dataset.CreateMynahDataset(db.NewContext(), &newDataset); err != nil {
			return err
		}

		return dataset.CreateMynahICDatasetVersion(tx, &dataset.MynahICDatasetVersion{
			DatasetVersionId: types.NewMynahUuid(),
			DatasetId:        newDataset.DatasetId,
			DateCreated:      newDataset.DateCreated,
			CreatedBy:        appCtx.User.UserId,
		})
	})
	if err != nil {
		log.Info("DatasetCreate failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	ctx.JSON(http.StatusOK, &newDataset)
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

// DatasetUploadFile uploads a file and adds to a dataset
func DatasetUploadFile(ctx *gin.Context) {
	appCtx := getAppContext(ctx)
	dbCtx := db.NewContext()

	// check that the dataset exists
	datasetId := types.MynahUuid(ctx.Param("dataset_id"))
	_, found, err := dataset.GetMynahDataset(dbCtx, datasetId)
	if err != nil {
		log.Info("DatasetUploadFile failed to get dataset: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	if !found {
		log.Info("DatasetUploadFile failed, no such dataset: %s", datasetId)
		ctx.Status(http.StatusNotFound)
		return
	}

	// check that the version exists
	versionId := types.MynahUuid(ctx.Param("version_id"))
	_, found, err = dataset.GetMynahICDatasetVersion(dbCtx, datasetId, versionId)
	if err != nil {
		log.Info("DatasetUploadFile failed to get dataset version: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	if !found {
		log.Info("DatasetUploadFile failed, no such dataset version: %s", versionId)
		ctx.Status(http.StatusNotFound)
		return
	}

	file, err := ctx.FormFile("file")
	if err != nil {
		log.Info("DatasetUploadFile failed: %s", err)
		ctx.Status(http.StatusBadRequest)
		return
	}

	mynahFile := models.MynahFile{
		FileId:      types.NewMynahUuid(),
		Name:        filepath.Base(file.Filename),
		DateCreated: time.Now().Unix(),
		CreatedBy:   appCtx.User.UserId,
	}

	err = db.NewContext().NewTransaction(func(tx *db.Context) error {
		// create the mynah file
		if err = models.CreateMynahFile(tx, &mynahFile); err != nil {
			return err
		}

		// create the dataset version reference
		//TODO

		return ctx.SaveUploadedFile(file, filepath.Join(settings.GlobalSettings.StorageSettings.LocalPath, string(mynahFile.FileId)))
	})
	if err != nil {
		log.Info("DatasetUploadFile failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}
}
