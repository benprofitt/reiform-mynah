// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"reiform.com/mynah/model"
)

// DBProvider Defines the interface that database clients must implement
type DBProvider interface {
	// GetUserForAuth Get a user by uuid or return an error
	GetUserForAuth(model.MynahUuid) (*model.MynahUser, error)
	// GetUser Get a user other than self (must be admin -- second argument)
	GetUser(model.MynahUuid, *model.MynahUser) (*model.MynahUser, error)
	// GetFile get a file from the database
	GetFile(model.MynahUuid, *model.MynahUser) (*model.MynahFile, error)
	// GetFiles get multiple files by id
	GetFiles([]model.MynahUuid, *model.MynahUser) (map[model.MynahUuid]*model.MynahFile, error)
	// GetICDataset get a dataset from the database
	GetICDataset(model.MynahUuid, *model.MynahUser) (*model.MynahICDataset, error)
	// GetODDataset get a dataset from the database
	GetODDataset(model.MynahUuid, *model.MynahUser) (*model.MynahODDataset, error)
	// GetICDatasets get multiple ic datasets from the database
	GetICDatasets([]model.MynahUuid, *model.MynahUser) (map[model.MynahUuid]*model.MynahICDataset, error)
	// GetODDatasets get multiple oc datasets from the database
	GetODDatasets([]model.MynahUuid, *model.MynahUser) (map[model.MynahUuid]*model.MynahODDataset, error)
	// ListUsers list all users, arg is requestor
	ListUsers(*model.MynahUser) ([]*model.MynahUser, error)
	// ListFiles list all files, arg is requestor
	ListFiles(*model.MynahUser) ([]*model.MynahFile, error)
	// ListICDatasets list all datasets, arg is requestor
	ListICDatasets(*model.MynahUser) ([]*model.MynahICDataset, error)
	// ListODDatasets list all datasets, arg is requestor
	ListODDatasets(*model.MynahUser) ([]*model.MynahODDataset, error)
	// CreateUser create a new user (second argument is the creator --must be admin)
	CreateUser(*model.MynahUser, func(*model.MynahUser) error) (*model.MynahUser, error)
	// CreateFile create a new file, arg is creator
	CreateFile(*model.MynahUser, func(*model.MynahFile) error) (*model.MynahFile, error)
	// CreateICDataset create a new dataset
	CreateICDataset(*model.MynahUser, func(*model.MynahICDataset) error) (*model.MynahICDataset, error)
	// CreateODDataset create a new dataset
	CreateODDataset(*model.MynahUser, func(*model.MynahODDataset) error) (*model.MynahODDataset, error)
	// UpdateUser update a user in the database. First arg is user to update, second is requestor, remaining
	//are keys to update.
	UpdateUser(*model.MynahUser, *model.MynahUser, ...string) error
	// UpdateICDataset update a dataset
	UpdateICDataset(*model.MynahICDataset, *model.MynahUser, ...string) error
	// UpdateODDataset update a dataset
	UpdateODDataset(*model.MynahODDataset, *model.MynahUser, ...string) error
	// UpdateFile updates a file
	UpdateFile(*model.MynahFile, *model.MynahUser, ...string) error
	// DeleteUser delete a user in the database, second arg is requestor
	DeleteUser(model.MynahUuid, *model.MynahUser) error
	// DeleteFile delete a file in the database, second arg is requestor
	DeleteFile(model.MynahUuid, *model.MynahUser) error
	// DeleteICDataset delete a dataset
	DeleteICDataset(model.MynahUuid, *model.MynahUser) error
	// DeleteODDataset delete a dataset
	DeleteODDataset(model.MynahUuid, *model.MynahUser) error
	// Close close the client connection on shutdown
	Close()
}
