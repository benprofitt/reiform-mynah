// Copyright (c) 2023 by Reiform. All Rights Reserved.

package db

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
