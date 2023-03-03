// Copyright (c) 2023 by Reiform. All Rights Reserved.

package middleware

import (
	"github.com/gin-gonic/gin"
	"reiform.com/mynah-api/models"
	mynah_context "reiform.com/mynah-api/services/context"
)

func Auth(c *gin.Context) {
	c.Set(appContextKey, &mynah_context.Context{
		User: &models.MynahUser{
			UserId: "none",
		},
	})
	// if not authenticated, Abort()
}
