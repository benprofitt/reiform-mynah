// Copyright (c) 2023 by Reiform. All Rights Reserved.

package middleware

import (
	"github.com/gin-gonic/gin"
	"reiform.com/mynah-api/services/log"
	"time"
)

func Logger(c *gin.Context) {
	start := time.Now()
	reqPath := c.Request.URL.Path
	raw := c.Request.URL.RawQuery

	c.Next()

	duration := time.Since(start)

	clientIP := c.ClientIP()
	method := c.Request.Method
	statusCode := c.Writer.Status()
	bodySize := c.Writer.Size()
	if raw != "" {
		reqPath = reqPath + "?" + raw
	}

	logHandler := log.Info

	if c.Writer.Status() >= 500 {
		logHandler = log.Error
	}

	logHandler("[%s] \"%s %s\" %d %d (%v)",
		clientIP,
		method,
		reqPath,
		statusCode,
		bodySize,
		duration)
}
