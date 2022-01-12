package db

import (
  _ "github.com/mattn/go-sqlite3"
  "reiform.com/mynah/model"
  "reiform.com/mynah/settings"
  "database/sql"
  "os"
  "log"
  "errors"
)

//check if the local mynah database already exists
func localDBExists(path *string) bool {
  if _, err := os.Stat(*path); err == nil {
    return true
  } else if errors.Is(err, os.ErrNotExist) {
    return false
  } else {
    log.Fatalf("failed to identify whether database already exists: %s", err)
    return false
  }
}

//prepare and execute a sql statement
func (d *localDB) localPrepareExec(statement string) error {
  if statement, err := d.db.Prepare(statement); err == nil {
    if _, execErr := statement.Exec(); execErr != nil {
      return execErr
    }
    return nil
	} else {
    return err
  }
}

//create the mynah local database
func createLocalDatabase(path *string) (*localDB, error) {
  //create the path
  if file, err := os.Create(*path); err == nil {
    file.Close()
  } else {
    return nil, err
  }

  //open the database
  sqlDB, err := sql.Open("sqlite3", *path)
  if err != nil {
    return nil, err
  }

  var ldb localDB
  ldb.db = sqlDB

  //create user table
  if userTableErr := ldb.localPrepareExec(createUserTableSQL); userTableErr != nil {
    return nil, userTableErr
  }

  //TODO create the projects table
  return &ldb, nil
}

//create a new local db instance
func newLocalDB(mynahSettings *settings.MynahSettings) (*localDB, error) {
  //only create database if it doesn't exist
  if !localDBExists(&mynahSettings.DBSettings.LocalPath) {
    db, err := createLocalDatabase(&mynahSettings.DBSettings.LocalPath)
    if err != nil {
      return nil, err
    }
    log.Printf("created local database %s", mynahSettings.DBSettings.LocalPath)
    return db, nil
  } else {
    //open the database
    sqlDB, err := sql.Open("sqlite3", mynahSettings.DBSettings.LocalPath)
    if err != nil {
      return nil, err
    }
    return &localDB{
      db: sqlDB,
    }, nil
  }
}

//Get a user by uuid or return an error
func (d *localDB) GetUser(*string) (*model.MynahUser, error) {
  return nil, nil
}

//get a project by id or return an error
func (d *localDB) GetProject(*string, *model.MynahUser) (*model.MynahProject, error) {
  return nil, nil
}

//update a user in the database
func (d *localDB) UpdateUser(*model.MynahUser) error {
  return nil
}

//update a project in the database
func (d *localDB) UpdateProject(*model.MynahProject, *model.MynahUser) error {
  return nil
}

//delete a user in the database
func (d *localDB) DeleteUser(*string) error {
  return nil
}

//delete a project in the database
func (d *localDB) DeleteProject(*string, *model.MynahUser) error {
  return nil
}

//close the client connection on shutdown
func (d *localDB) Close() {

}
