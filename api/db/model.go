// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"reiform.com/mynah/model"
)

//Defines the interface that database clients must implement
type DBProvider interface {
	//Get a user by uuid or return an error
	GetUserForAuth(*string) (*model.MynahUser, error)
	//Get a user other than self (must be admin -- second argument)
	GetUser(*string, *model.MynahUser) (*model.MynahUser, error)
	//get a project by id or return an error, second arg is requestor
	GetProject(*string, *model.MynahUser) (*model.MynahProject, error)
	//get a file from the database
	GetFile(*string, *model.MynahUser) (*model.MynahFile, error)
	//get multiple files by id
	GetFiles([]string, *model.MynahUser) ([]*model.MynahFile, error)
	//get a dataset from the database
	GetDataset(*string, *model.MynahUser) (*model.MynahDataset, error)
	//get a dataset from the database
	GetICDataset(*string, *model.MynahUser) (*model.MynahICDataset, error)
	//list all users, arg is requestor
	ListUsers(*model.MynahUser) ([]*model.MynahUser, error)
	//list all projects, arg is requestor
	ListProjects(*model.MynahUser) ([]*model.MynahProject, error)
	//list all files, arg is requestor
	ListFiles(*model.MynahUser) ([]*model.MynahFile, error)
	//list all datasets, arg is requestor
	ListDatasets(*model.MynahUser) ([]*model.MynahDataset, error)
	//list all datasets, arg is requestor
	ListICDatasets(*model.MynahUser) ([]*model.MynahICDataset, error)
	//create a new user (second argument is the creator --must be admin)
	CreateUser(*model.MynahUser, *model.MynahUser) error
	//create a new project, second arg is creator
	CreateProject(*model.MynahProject, *model.MynahUser) error
	//create a new file, second arg is creator
	CreateFile(*model.MynahFile, *model.MynahUser) error
	//create a new dataset
	CreateDataset(*model.MynahDataset, *model.MynahUser) error
	//create a new dataset
	CreateICDataset(*model.MynahICDataset, *model.MynahUser) error
	//update a user in the database. First arg is user to update, second is requestor, remaining
	//are keys to update.
	UpdateUser(*model.MynahUser, *model.MynahUser, ...string) error
	//update a project in the database. First arg is uuid of project to update, second is requestor, remaining
	//are keys to update
	UpdateProject(*model.MynahProject, *model.MynahUser, ...string) error
	//update a dataset
	UpdateDataset(*model.MynahDataset, *model.MynahUser, ...string) error
	//update a dataset
	UpdateICDataset(*model.MynahICDataset, *model.MynahUser, ...string) error
	//delete a user in the database, second arg is requestor
	DeleteUser(*string, *model.MynahUser) error
	//delete a project in the database, second arg is requestor
	DeleteProject(*string, *model.MynahUser) error
	//delete a file in the database, second arg is requestor
	DeleteFile(*string, *model.MynahUser) error
	//delete a dataset
	DeleteDataset(*string, *model.MynahUser) error
	//delete a dataset
	DeleteICDataset(*string, *model.MynahUser) error
	//close the client connection on shutdown
	Close()
}
