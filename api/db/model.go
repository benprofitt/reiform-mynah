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
	//Get a user other than self (must be admin)
	GetUser(*string, *model.MynahUser) (*model.MynahUser, error)
	//get a project by id or return an error
	GetProject(*string, *model.MynahUser) (*model.MynahProject, error)
	//list all users
	ListUsers(*model.MynahUser) ([]*model.MynahUser, error)
	//list all projects
	ListProjects(*model.MynahUser) ([]*model.MynahProject, error)
	//create a new user (second argument is the creator --must be admin)
	CreateUser(*model.MynahUser, *model.MynahUser) error
	//create a new project
	CreateProject(*model.MynahProject, *model.MynahUser) error
	//update a user in the database
	UpdateUser(*model.MynahUser, *model.MynahUser) error
	//update a project in the database
	UpdateProject(*model.MynahProject, *model.MynahUser) error
	//delete a user in the database
	DeleteUser(*string, *model.MynahUser) error
	//delete a project in the database
	DeleteProject(*string, *model.MynahUser) error
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
