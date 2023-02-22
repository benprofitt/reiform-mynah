// Copyright (c) 2023 by Reiform. All Rights Reserved.

package db

import (
	"errors"
	"fmt"
	_ "github.com/mattn/go-sqlite3"
	"os"
	"path/filepath"
	"reiform.com/mynah-api/models/db/migrations"
	"reiform.com/mynah-api/services/log"
	"reiform.com/mynah-api/settings"
	"xorm.io/xorm"
)

var (
	coreEngine *xorm.Engine
	allTables  []interface{}
)

//create the database file if it doesn't exist return
func checkDBFile(path string) (exists bool, err error) {
	if _, err := os.Stat(path); err == nil {
		return true, nil
	} else if errors.Is(err, os.ErrNotExist) {
		if file, err := os.Create(filepath.Clean(path)); err == nil {
			return false, file.Close()
		} else {
			return false, err
		}
	} else {
		return false, fmt.Errorf("failed to identify whether database already exists: %s", err)
	}
}

// RegisterTables adds tables to be used when initializing a new database
func RegisterTables(tables ...interface{}) {
	allTables = append(allTables, tables...)
}

// StartDBEnvironment creates the db environment
func StartDBEnvironment() error {
	//check if the database file has been created
	exists, err := checkDBFile(settings.GlobalSettings.DBSettings.LocalPath)
	if err != nil {
		return err
	}

	//create the gorm engine
	coreEngine, err = xorm.NewEngine("sqlite3", settings.GlobalSettings.DBSettings.LocalPath)
	if err != nil {
		return err
	}

	if !exists {
		if err = coreEngine.CreateTables(allTables...); err != nil {
			return err
		}

		if err = coreEngine.Sync2(allTables...); err != nil {
			return err
		}

		log.Warn("created local database %s", settings.GlobalSettings.DBSettings.LocalPath)
	}

	// apply migrations
	return migrations.Migrate(coreEngine)
}
