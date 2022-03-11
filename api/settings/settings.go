// Copyright (c) 2022 by Reiform. All Rights Reserved.

package settings

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"
	"reiform.com/mynah/log"
	"strings"
)

type DBSetting string

const (
	Local    DBSetting = "local"
	External DBSetting = "external"
)

// MynahDBSettings defines settings for the database
type MynahDBSettings struct {
	//the database configuration (either external/local)
	Type DBSetting `json:"type"`
	//path to store the local database
	LocalPath string `json:"local_path"`
	//the number of organizations to create on startup
	InitialOrgCount int `json:"initial_org_count"`
}

// MynahAuthSettings defines settings for authentication
type MynahAuthSettings struct {
	//the path to the pem key for JWT validation and generation
	PemFilePath string `json:"pem_file_path"`
	//the http header containing the jwt
	JwtHeader string `json:"jwt_header"`
}

// MynahStorageSettings defines settings for storage
type MynahStorageSettings struct {
	//whether users can load/save data in s3
	S3Storage bool `json:"s3_storage"`
	//the path to store data to locally
	LocalPath string `json:"local_path"`
	//the max upload size
	MaxUpload int64 `json:"max_upload"`
}

// MynahPythonSettings defines configuration settings for python
type MynahPythonSettings struct {
	//the path where python modules are stored
	ModulePath string `json:"module_path"`
	//the top level module name
	ModuleName string `json:"module_name"`
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
	//the prefix for api paths
	ApiPrefix string `json:"api_prefix"`
	//prefix for static resources
	StaticPrefix string `json:"static_prefix"`
	//the folder containing static web resources
	StaticResourcesPath string `json:"static_resources_path"`
	//whether read access can be unauthenticated
	UnauthReadAccess bool `json:"unauth_read_access"`
	//the port to listen for requests on
	Port int `json:"port"`
	//origins to allow
	CORSAllowOrigin string `json:"cors_allow_origin"`
	//settings groups
	DBSettings      MynahDBSettings      `json:"db_settings"`
	AuthSettings    MynahAuthSettings    `json:"auth_settings"`
	StorageSettings MynahStorageSettings `json:"storage_settings"`
	PythonSettings  MynahPythonSettings  `json:"python_settings"`
	AsyncSettings   MynahAsyncSettings   `json:"async_settings"`
	IPCSettings     MynahIPCSettings     `json:"ipc_settings"`
}

// DefaultSettings get the default settings
func DefaultSettings() *MynahSettings {
	return &MynahSettings{
		ApiPrefix:           "/api/v1",
		StaticPrefix:        "/mynah/",
		StaticResourcesPath: "./static/",
		UnauthReadAccess:    false,
		Port:                8080,
		CORSAllowOrigin:     "*",
		DBSettings: MynahDBSettings{
			Type:            "local",
			LocalPath:       "data/mynah_local.db",
			InitialOrgCount: 1,
		},
		AuthSettings: MynahAuthSettings{
			PemFilePath: "data/auth.pem",
			JwtHeader:   "api-key",
		},
		StorageSettings: MynahStorageSettings{
			S3Storage: true,
			LocalPath: `data/tmp`,
			MaxUpload: 100 * 1024 * 1024 * 1024,
		},
		PythonSettings: MynahPythonSettings{
			ModulePath: "./python",
			ModuleName: "mynah",
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

//write the default settings to a file
func generateSettings(path *string) {
	m := DefaultSettings()

	dirPath := strings.TrimSuffix(*path, filepath.Base(*path))

	//create the base directory if it doesn't exist
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		log.Errorf("failed to create directory: %s", dirPath)
		return
	}

	//write to file
	if json, jsonErr := json.MarshalIndent(m, "", "  "); jsonErr == nil {
		if ioErr := ioutil.WriteFile(*path, json, 0600); ioErr != nil {
			log.Errorf("failed to write default settings: %s", ioErr)
		}
	} else {
		log.Errorf("failed to generate default settings: %s", jsonErr)
	}
}

//check whether a settings file exists
func settingsExist(path *string) bool {
	if _, err := os.Stat(*path); err == nil {
		return true
	} else if errors.Is(err, os.ErrNotExist) {
		return false
	} else {
		log.Fatalf("failed to identify whether settings file already exists: %s", err)
		return false
	}
}

// LoadSettings Load Mynah settings from a file
func LoadSettings(path *string) (*MynahSettings, error) {
	if !settingsExist(path) {
		//generate default settings
		generateSettings(path)
		log.Warnf("wrote new settings file to %s", *path)
	}

	//read in the settings file from the local path
	if file, fileErr := ioutil.ReadFile(*path); fileErr == nil {
		var settings MynahSettings
		if jsonErr := json.Unmarshal([]byte(file), &settings); jsonErr == nil {
			return &settings, nil
		} else {
			return nil, jsonErr
		}
	} else {
		return nil, fileErr
	}
}
