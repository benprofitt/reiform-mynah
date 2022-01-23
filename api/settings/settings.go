package settings

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"log"
	"os"
)

type DBSetting string

const (
	Local    DBSetting = "local"
	External DBSetting = "external"
)

//defines settings for the database
type MynahDBSettings struct {
	//the database configuration (either external/local)
	Type DBSetting `json:"type"`
	//path to store the local database
	LocalPath string `json:"local_path"`
	//the number of organizations to create on startup
	InitialOrgCount int `json:"initial_org_count"`
}

//defines settings for authentication
type MynahAuthSettings struct {
	//the path to the pem key for JWT validation and generation
	PemFilePath string `json:"pem_file_path"`
	//the http header containing the jwt
	JwtHeader string `json:"jwt_header"`
}

//defines settings for storage
type MynahStorageSettings struct {
	//whether users can load/save data in s3
	S3Storage bool `json:"s3_storage"`
	//the path to store data to locally
	LocalPath string `json:"local_path"`
	//the max upload size
	MaxUpload int64 `json:"max_upload"`
}

//defines configuration settings for python
type MynahPythonSettings struct {
	//the path where python modules are stored
	ModulePath string `json:"module_path"`
}

//defines configuration settings for async task engine
type MynahAsyncSettings struct {
	//how many async workers to use
	Workers int `json:"workers"`
	//The size of the async buffer
	BufferSize int `json:"buffer_size"`
}

//Defines various settings for the application
type MynahSettings struct {
	//the prefix for api paths
	ApiPrefix string `json:"api_prefix"`
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
}

//get the default settings
func DefaultSettings() *MynahSettings {
	return &MynahSettings{
		ApiPrefix:        "/api/v1",
		UnauthReadAccess: false,
		Port:             8080,
		CORSAllowOrigin:  "*",
		DBSettings: MynahDBSettings{
			Type:            "local",
			LocalPath:       "mynah_local.db",
			InitialOrgCount: 1,
		},
		AuthSettings: MynahAuthSettings{
			PemFilePath: "auth.pem",
			JwtHeader:   "api-key",
		},
		StorageSettings: MynahStorageSettings{
			S3Storage: true,
			LocalPath: `tmp`,
			MaxUpload: 100 * 1024 * 1024 * 1024,
		},
		PythonSettings: MynahPythonSettings{
			ModulePath: "./python",
		},
		AsyncSettings: MynahAsyncSettings{
			Workers: 3,
			BufferSize: 256,
		},
	}
}

//write the default settings to a file
func generateSettings(path *string) {
	m := DefaultSettings()

	//write to file
	if json, jsonErr := json.MarshalIndent(m, "", "  "); jsonErr == nil {
		if ioErr := ioutil.WriteFile(*path, json, 0644); ioErr != nil {
			log.Printf("failed to write default settings: %s", ioErr)
		}
	} else {
		log.Printf("failed to generate default settings: %s", jsonErr)
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

//Load Mynah settings from a file
func LoadSettings(path *string) (*MynahSettings, error) {
	if !settingsExist(path) {
		//generate default settings
		generateSettings(path)
		log.Printf("wrote new settings file to %s", *path)
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
