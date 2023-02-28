// Copyright (c) 2023 by Reiform. All Rights Reserved.

package settings

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"
	"reiform.com/mynah-api/services/log"
	"strings"
)

// MynahApplicationVersion is the current application version
const MynahApplicationVersion = "0.1.0"

// MynahDBSettings defines settings for the database
type MynahDBSettings struct {
	//path to store the local database
	LocalPath string `json:"local_path"`
}

// MynahStorageSettings defines settings for storage
type MynahStorageSettings struct {
	//the path to store data to locally
	LocalPath string `json:"local_path"`
	//the path to where models are stored
	ModelsPath string `json:"models_path"`
}

// MynahPythonSettings defines configuration settings for python
type MynahPythonSettings struct {
	//python executable command
	PythonExecutable string `json:"python_executable"`
}

// MynahAsyncSettings defines configuration settings for async task engine
type MynahAsyncSettings struct {
	//how many async workers to use
	Workers int `json:"workers"`
	//The size of the async buffer
	BufferSize int `json:"buffer_size"`
}

// MynahIPCSettings defines settings for ipc
type MynahIPCSettings struct {
	//the socket address to communicate on
	SocketAddr string `json:"socket_addr"`
}

// MynahSettings Defines various settings for the application
type MynahSettings struct {
	ApiPrefix           string   `json:"api_prefix"`
	StaticPrefix        string   `json:"static_prefix"`
	StaticResourcesPath string   `json:"static_resources_path"`
	UnauthReadAccess    bool     `json:"unauth_read_access"`
	Port                int      `json:"port"`
	CORSAllowHeaders    []string `json:"cors_allow_headers"`
	CORSAllowOrigins    []string `json:"cors_allow_origins"`
	CORSAllowMethods    []string `json:"cors_allow_methods"`
	DefaultPageSize     int      `json:"page_size"`
	//settings groups
	DBSettings      MynahDBSettings      `json:"db_settings"`
	StorageSettings MynahStorageSettings `json:"storage_settings"`
	PythonSettings  MynahPythonSettings  `json:"python_settings"`
	AsyncSettings   MynahAsyncSettings   `json:"async_settings"`
	IPCSettings     MynahIPCSettings     `json:"ipc_settings"`
}

var GlobalSettings *MynahSettings

// defaultSettings get the default settings
func defaultSettings() *MynahSettings {
	return &MynahSettings{
		ApiPrefix:           "/api/v1",
		StaticPrefix:        "/mynah/",
		StaticResourcesPath: "./static/",
		UnauthReadAccess:    false,
		Port:                8080,
		CORSAllowHeaders:    []string{"*"},
		CORSAllowOrigins:    []string{"*"},
		CORSAllowMethods:    []string{"POST", "GET"},
		DefaultPageSize:     20,
		DBSettings: MynahDBSettings{
			LocalPath: "data/mynah_local.db",
		},
		StorageSettings: MynahStorageSettings{
			LocalPath:  `data/tmp`,
			ModelsPath: "python/models",
		},
		PythonSettings: MynahPythonSettings{
			PythonExecutable: "./python/mynah.py",
		},
		AsyncSettings: MynahAsyncSettings{
			Workers:    3,
			BufferSize: 256,
		},
		IPCSettings: MynahIPCSettings{
			SocketAddr: "/tmp/mynah.sock",
		},
	}
}

//check whether a settings file exists
func settingsExist(path *string) bool {
	if _, err := os.Stat(*path); err == nil {
		return true
	} else if errors.Is(err, os.ErrNotExist) {
		return false
	} else {
		log.Fatal("failed to identify whether settings file already exists: %s", err)
		return false
	}
}

// Load loads the settings for the app from a file (or default)
func Load(path *string) error {
	if !settingsExist(path) {
		settings := defaultSettings()

		dirPath := strings.TrimSuffix(*path, filepath.Base(*path))

		//create the base directory if it doesn't exist
		if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
			return err
		}

		jsonContents, err := json.MarshalIndent(settings, "", "  ")
		if err != nil {
			return err
		}

		if ioErr := ioutil.WriteFile(*path, jsonContents, 0600); ioErr != nil {
			return err
		}

		GlobalSettings = settings
		log.Warn("wrote new settings file to %s", *path)

	} else {
		file, err := ioutil.ReadFile(*path)
		if err != nil {
			return err
		}

		var settings MynahSettings
		if err := json.Unmarshal(file, &settings); err != nil {
			return err
		}
		GlobalSettings = &settings
	}

	return nil
}
