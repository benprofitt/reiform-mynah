// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"reiform.com/mynah/model"
)

// DBProvider Defines the interface that database clients must implement
type DBProvider interface {
	// GetUserForAuth Get a user by uuid or return an error
	GetUserForAuth(*string) (*model.MynahUser, error)
	// GetUser Get a user other than self (must be admin -- second argument)
	GetUser(*string, *model.MynahUser) (*model.MynahUser, error)
	// GetProject get a project by id or return an error, second arg is requestor
	GetProject(*string, *model.MynahUser) (*model.MynahProject, error)
	// GetICProject get a project by id or return an error, second arg is requestor
	GetICProject(*string, *model.MynahUser) (*model.MynahICProject, error)
	// GetFile get a file from the database
	GetFile(*string, *model.MynahUser) (*model.MynahFile, error)
	// GetFiles get multiple files by id
	GetFiles([]string, *model.MynahUser) ([]*model.MynahFile, error)
	// GetDataset get a dataset from the database
	GetDataset(*string, *model.MynahUser) (*model.MynahDataset, error)
	// GetICDataset get a dataset from the database
	GetICDataset(*string, *model.MynahUser) (*model.MynahICDataset, error)
	// GetICDatasets get multiple ic datasets from the database
	GetICDatasets([]string, *model.MynahUser) ([]*model.MynahICDataset, error)
	// ListUsers list all users, arg is requestor
	ListUsers(*model.MynahUser) ([]*model.MynahUser, error)
	// ListProjects list all projects, arg is requestor
	ListProjects(*model.MynahUser) ([]*model.MynahProject, error)
	// ListICProjects list all projects, arg is requestor
	ListICProjects(*model.MynahUser) ([]*model.MynahICProject, error)
	// ListFiles list all files, arg is requestor
	ListFiles(*model.MynahUser) ([]*model.MynahFile, error)
	// ListDatasets list all datasets, arg is requestor
	ListDatasets(*model.MynahUser) ([]*model.MynahDataset, error)
	// ListICDatasets list all datasets, arg is requestor
	ListICDatasets(*model.MynahUser) ([]*model.MynahICDataset, error)
	// CreateUser create a new user (second argument is the creator --must be admin)
	CreateUser(*model.MynahUser, func(*model.MynahUser)) (*model.MynahUser, error)
	// CreateProject create a new project, arg is creator
	CreateProject(*model.MynahUser, func(*model.MynahProject)) (*model.MynahProject, error)
	// CreateICProject create a new project, arg is creator
	CreateICProject(*model.MynahUser, func(*model.MynahICProject)) (*model.MynahICProject, error)
	//create a new file, arg is creator
	CreateFile(*model.MynahUser, func(*model.MynahFile) error) (*model.MynahFile, error)
	//create a new dataset
	CreateDataset(*model.MynahUser, func(*model.MynahDataset)) (*model.MynahDataset, error)
	// CreateICDataset create a new dataset
	CreateICDataset(*model.MynahUser, func(*model.MynahICDataset)) (*model.MynahICDataset, error)
	// UpdateUser update a user in the database. First arg is user to update, second is requestor, remaining
	//are keys to update.
	UpdateUser(*model.MynahUser, *model.MynahUser, ...string) error
	// UpdateProject update a project in the database. First arg is uuid of project to update, second is requestor, remaining
	//are keys to update
	UpdateProject(*model.MynahProject, *model.MynahUser, ...string) error
	// UpdateICProject update a project in the database. First arg is uuid of project to update, second is requestor, remaining
	//are keys to update
	UpdateICProject(*model.MynahICProject, *model.MynahUser, ...string) error
	// UpdateDataset update a dataset
	UpdateDataset(*model.MynahDataset, *model.MynahUser, ...string) error
	// UpdateICDataset update a dataset
	UpdateICDataset(*model.MynahICDataset, *model.MynahUser, ...string) error
	// DeleteUser delete a user in the database, second arg is requestor
	DeleteUser(*string, *model.MynahUser) error
	// DeleteProject delete a project in the database, second arg is requestor
	DeleteProject(*string, *model.MynahUser) error
	// DeleteICProject delete a project in the database, second arg is requestor
	DeleteICProject(*string, *model.MynahUser) error
	// DeleteFile delete a file in the database, second arg is requestor
	DeleteFile(*string, *model.MynahUser) error
	// DeleteDataset delete a dataset
	DeleteDataset(*string, *model.MynahUser) error
	// DeleteICDataset delete a dataset
	DeleteICDataset(*string, *model.MynahUser) error
	// Close close the client connection on shutdown
	Close()
}
