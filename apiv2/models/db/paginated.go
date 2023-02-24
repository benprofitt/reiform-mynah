// Copyright (c) 2023 by Reiform. All Rights Reserved.

package db

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"reiform.com/mynah-api/settings"
	"strconv"
)

// Paginated defines some paginated database result
type Paginated[T any] struct {
	Page     int   `json:"page"`
	PageSize int   `json:"page_size"`
	Total    int64 `json:"total"`
	Contents []T   `json:"contents"`
}

// PaginationOptions defines the current paging info for a request
type PaginationOptions struct {
	Page     int
	PageSize int
}

// GetPaginationOptions parses pagination options from gin context
func GetPaginationOptions(ctx *gin.Context) (*PaginationOptions, error) {
	pageS := ctx.DefaultQuery("page", "0")
	pageSizeS := ctx.DefaultQuery("page_size", fmt.Sprintf("%d", settings.GlobalSettings.DefaultPageSize))
	page, err := strconv.Atoi(pageS)
	if err != nil {
		return nil, fmt.Errorf("faled to parse 'page' as int: %w", err)
	}
	pageSize, err := strconv.Atoi(pageSizeS)
	if err != nil {
		return nil, fmt.Errorf("faled to parse 'page_size' as int: %w", err)
	}

	return &PaginationOptions{
		Page:     page,
		PageSize: pageSize,
	}, nil
}
