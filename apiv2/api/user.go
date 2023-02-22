// Copyright (c) 2023 by Reiform. All Rights Reserved.

package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
	"reiform.com/mynah-api/models"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/services/log"
	"reiform.com/mynah-api/types"
)

// UserCreateBody defines the request body for UserCreate
type UserCreateBody struct {
	NameFirst string `json:"name_first" binding:"required"`
	NameLast  string `json:"name_last" binding:"required"`
}

// UserCreate creates a new user
func UserCreate(ctx *gin.Context) {
	var body UserCreateBody
	if err := ctx.ShouldBindJSON(&body); err != nil {
		log.Info("UserCreate failed: %s", err)
		ctx.Status(http.StatusBadRequest)
		return
	}

	newUser := &models.MynahUser{
		UserId:    types.NewMynahUuid(),
		NameFirst: body.NameFirst,
		NameLast:  body.NameLast,
	}
	if err := models.CreateMynahUser(db.NewContext(), newUser); err != nil {
		log.Info("UserCreate failed: %s", err)
		ctx.Status(http.StatusInternalServerError)
		return
	}

	ctx.JSON(http.StatusOK, newUser)
}
