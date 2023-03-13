// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
	"path/filepath"
	"reiform.com/mynah-api/api/middleware"
	dataset_model "reiform.com/mynah-api/models/dataset"
	"reiform.com/mynah-api/models/db"
	file_model "reiform.com/mynah-api/models/file"
	"reiform.com/mynah-api/models/types"
	"reiform.com/mynah-api/services/log"
	"reiform.com/mynah-api/settings"
	"time"
)

// DatasetCreateBody defines the request body for DatasetCreate
type DatasetCreateBody struct {
	DatasetName string                         `json:"dataset_name" binding:"required"`
	DatasetType dataset_model.MynahDatasetType `json:"dataset_type" binding:"required,mynah_dataset_type"`
}

// DatasetClassesBody defines the request body for DatasetClasses
type DatasetClassesBody struct {
	Assignments map[types.MynahUuid]dataset_model.MynahClassName `json:"assignments" binding:"required"`
}

// DatasetCreate creates a new dataset
func DatasetCreate(ctx *gin.Context) {
	appCtx := middleware.GetAppContext(ctx)

	var body DatasetCreateBody
	if err := ctx.ShouldBindJSON(&body); err != nil {
		log.Info("DatasetCreate failed: %s", err)
		ctx.Status(http.StatusBadRequest)
		return
	}

	newDataset := dataset_model.MynahDataset{
		DatasetId:    types.NewMynahUuid(),
		DatasetName:  body.DatasetName,
		DateCreated:  time.Now().Unix(),
		DateModified: time.Now().Unix(),
		DatasetType:  body.DatasetType,
		CreatedBy:    "no-user", // TODO no user ownership currently
	}

	err := db.NewContext().NewTransaction(func(tx *db.Context) error {
		if err := dataset_model.CreateMynahDataset(db.NewContext(), &newDataset); err != nil {
			return err
		}

		return dataset_model.CreateMynahICDatasetVersion(tx, &dataset_model.MynahICDatasetVersion{
			DatasetVersionId: types.NewMynahUuid(),
			DatasetId:        newDataset.DatasetId,
			DateCreated:      newDataset.DateCreated,
			Mean:             make([]float64, 0),
			StdDev:           make([]float64, 0),
			TaskData:         make([]*dataset_model.MynahICProcessTaskData, 0),
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
	ctx.JSON(http.StatusOK, middleware.GetDatasetFromContext(ctx))
}

// DatasetList lists datasets
func DatasetList(ctx *gin.Context) {
	datasets, err := dataset_model.ListMynahDatasets(db.NewContext(), middleware.GetPaginationOptionsFromContext(ctx))
	if err != nil {
		log.Info("DatasetList failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	ctx.JSON(http.StatusOK, datasets)
}

// DatasetUploadFile uploads a file and adds to a dataset
func DatasetUploadFile(ctx *gin.Context) {
	appCtx := middleware.GetAppContext(ctx)

	file, err := ctx.FormFile("file")
	if err != nil {
		log.Info("DatasetUploadFile failed: %s", err)
		ctx.Status(http.StatusBadRequest)
		return
	}

	mynahFile := file_model.MynahFile{
		FileId:      types.NewMynahUuid(),
		Name:        filepath.Base(file.Filename),
		DateCreated: time.Now().Unix(),
		CreatedBy:   appCtx.User.UserId,
	}

	err = db.NewContext().NewTransaction(func(tx *db.Context) error {
		if err = file_model.CreateMynahFile(tx, &mynahFile); err != nil {
			return err
		}
		return ctx.SaveUploadedFile(file, filepath.Join(settings.GlobalSettings.StorageSettings.LocalPath, string(mynahFile.FileId)))
	})
	if err != nil {
		log.Info("DatasetUploadFile failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	// respond with the newly created file
	ctx.JSON(http.StatusOK, &mynahFile)
}

// DatasetVersionRefs returns the dataset version ordering
func DatasetVersionRefs(ctx *gin.Context) {
	// get the versions
	versions, err := dataset_model.ListMynahICDatasetVersionRefs(db.NewContext(), middleware.GetDatasetFromContext(ctx).DatasetId)
	if err != nil {
		log.Info("DatasetVersions failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}
	ctx.JSON(http.StatusOK, versions)
}

// DatasetVersionGet gets a specific version of a dataset
func DatasetVersionGet(ctx *gin.Context) {
	ctx.JSON(http.StatusOK, middleware.GetICDatasetVersionFromContext(ctx))
}

// DatasetVersionList lists dataset versions
func DatasetVersionList(ctx *gin.Context) {
	versions, err := dataset_model.ListMynahICDatasets(db.NewContext(),
		middleware.GetDatasetFromContext(ctx).DatasetId,
		middleware.GetPaginationOptionsFromContext(ctx))
	if err != nil {
		log.Info("DatasetVersionList failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}
	ctx.JSON(http.StatusOK, versions)
}

// DatasetClasses sets the classes for files in a dataset
func DatasetClasses(ctx *gin.Context) {
	var body DatasetClassesBody
	if err := ctx.ShouldBindJSON(&body); err != nil {
		log.Info("DatasetClasses failed: %s", err)
		ctx.Status(http.StatusBadRequest)
		return
	}

	if len(body.Assignments) == 0 {
		log.Info("DatasetClasses received empty class assignment list, ignoring")
		ctx.Status(http.StatusOK)
		return
	}

	err := file_model.AssignMynahICDatasetClasses(db.NewContext(),
		middleware.GetICDatasetVersionFromContext(ctx).DatasetVersionId,
		body.Assignments)
	if err != nil {
		log.Info("DatasetClasses failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	ctx.Status(http.StatusOK)
}
