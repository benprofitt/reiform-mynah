package db

import (
	"database/sql"
	_ "github.com/mattn/go-sqlite3"
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
	//list all users, arg is requestor
	ListUsers(*model.MynahUser) ([]*model.MynahUser, error)
	//list all projects, arg is requestor
	ListProjects(*model.MynahUser) ([]*model.MynahProject, error)
	//list all files, arg is requestor
	ListFiles(*model.MynahUser) ([]*model.MynahFile, error)
	//create a new user (second argument is the creator --must be admin)
	CreateUser(*model.MynahUser, *model.MynahUser) error
	//create a new project, second arg is creator
	CreateProject(*model.MynahProject, *model.MynahUser) error
	//create a new file, second arg is creator
	CreateFile(*model.MynahFile, *model.MynahUser) error
	//update a user in the database, second arg is requestor
	UpdateUser(*model.MynahUser, *model.MynahUser) error
	//update a project in the database, second arg is requestor
	UpdateProject(*model.MynahProject, *model.MynahUser) error
	//update a file in the database, second arg is requestor
	UpdateFile(*model.MynahFile, *model.MynahUser) error
	//delete a user in the database, second arg is requestor
	DeleteUser(*string, *model.MynahUser) error
	//delete a project in the database, second arg is requestor
	DeleteProject(*string, *model.MynahUser) error
	//delete a file in the database, second arg is requestor
	DeleteFile(*string, *model.MynahUser) error
	//close the client connection on shutdown
	Close()
}

//local database client adheres to DBProvider
type localDB struct {
	//the sqlite database client
	db *sql.DB
}

//external database client adheres to DBProvider
type externalDB struct {
}
