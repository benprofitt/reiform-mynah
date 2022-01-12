package db

import (
	"reiform.com/mynah/model"
	_ "github.com/mattn/go-sqlite3"
	"database/sql"
)

//Defines the interface that database clients must implement
type DBProvider interface {
	//Get a user by uuid or return an error
	GetUser(*string) (*model.MynahUser, error)
	//get a project by id or return an error
	GetProject(*string, *model.MynahUser) (*model.MynahProject, error)
	//update a user in the database
	UpdateUser(*model.MynahUser) error
	//update a project in the database
	UpdateProject(*model.MynahProject, *model.MynahUser) error
	//delete a user in the database
	DeleteUser(*string) error
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
