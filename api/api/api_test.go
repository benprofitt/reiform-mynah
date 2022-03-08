// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/test"
	"testing"
)

//setup and teardown
func TestMain(m *testing.M) {
	dirPath := "data"

	//create the base directory if it doesn't exist
	if err := os.MkdirAll(dirPath, os.ModePerm); err != nil {
		log.Fatalf("failed to create directory: %s", dirPath)
	}

	//run tests
	exitVal := m.Run()

	//remove generated
	if err := os.RemoveAll(dirPath); err != nil {
		log.Errorf("failed to clean up after tests: %s", err)
	}

	os.Exit(exitVal)
}

//create a new user, make an authenticated create user request
func makeCreateUserReq(user *model.MynahUser, jwt string, c *test.TestContext) error {
	var jsonBody = []byte(`{"name_first":"test", "name_last": "test"}`)

	//try to create a user
	req, reqErr := http.NewRequest("POST", filepath.Join(c.Settings.ApiPrefix, "admin", "user/create"), bytes.NewBuffer(jsonBody))
	if reqErr != nil {
		return reqErr
	}

	//include the content type
	req.Header.Add("Content-Type", "application/json")

	//make an http request
	return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
		if code != http.StatusOK {
			return fmt.Errorf("create user did not return a 200 status")
		}

		//check that the user was inserted into the database
		var res adminCreateUserResponse
		//check the body
		if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
			return fmt.Errorf("failed to decode response %s", err)
		}

		//check for the user in the database (as a known admin)
		dbUser, dbErr := c.DBProvider.GetUser(&res.User.Uuid, user)
		if dbErr != nil {
			return fmt.Errorf("new user not found in database %s", dbErr)
		}

		//verify same
		if dbUser.OrgId != user.OrgId {
			return fmt.Errorf("user from db (%v) not assigned same org id (%v)", dbUser.OrgId, user.OrgId)
		}
		return nil
	})
}

//Test admin endpoints
func TestAPIAdminEndpoints(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		//handle user creation endpoint
		c.Router.HandleAdminRequest("POST", "user/create", adminCreateUser(c.DBProvider, c.AuthProvider))

		//create as admin
		err := c.WithCreateUser(true, func(user *model.MynahUser, jwt string) error {
			return makeCreateUserReq(user, jwt, c)
		})
		if err != nil {
			return err
		}

		//create as non-admin
		err = c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			return makeCreateUserReq(user, jwt, c)
		})
		if err == nil {
			return errors.New("create user as non admin did not produce error as expected")
		}
		return nil
	})

	if err != nil {
		t.Fatalf("TestAPIAdminEndpoints error: %s", err)
	}
}

func TestFileGetEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		c.Router.HandleAdminRequest("POST", "user/create", adminCreateUser(c.DBProvider, c.AuthProvider))
		c.Router.HandleFileRequest("file")

		testContents := "test contents"
		expectedType := "text/plain; charset=utf-8"

		//create a user
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			//create a file
			return c.WithCreateFile(user, testContents, func(file *model.MynahFile) error {
				//make a request for the file
				req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, "file", file.Uuid), nil)
				if reqErr != nil {
					return reqErr
				}

				//make a request for the file
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("create user returned non-200: %v want %v", code, http.StatusOK)
					}

					if rr.Result().ContentLength != int64(len(testContents)) {
						t.Fatalf("file contents length served (%d) not the same as saved (%d)\n",
							rr.Result().ContentLength,
							len(testContents))
					}

					if rr.Result().Header.Get("Content-Type") != expectedType {
						t.Fatalf("file contents served (%s) not the same as saved (%s)\n",
							rr.Result().Header.Get("Content-Type"),
							expectedType)
					}
					return nil
				})
			})
		})
	})
	if err != nil {
		t.Fatalf("TestFileGetEndpoint error: %s", err)
	}
}

func TestAPIStartDiagnosisJobEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.ModulePath = "../../python"
	mynahSettings.PythonSettings.ModuleName = "mynah_test"

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		//create a user
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			//create a file
			return c.WithCreateFullICProject(user, func(project *model.MynahICProject) error {

				errChan := make(chan error)
				readyChan := make(chan struct{})

				//listen for websocket response
				go c.WebsocketListener(jwt, 1, readyChan, errChan, func(res string) error {
					//TODO check more
					if res == "{}" {
						return nil
					}
					return fmt.Errorf("unexpected response: %s", res)
				})

				//wait for the websocket server to be up
				<-readyChan

				//create the request body
				reqBody := startDiagnosisJobRequest{
					ProjectUuid: project.Uuid,
				}

				jsonBody, err := json.Marshal(reqBody)
				if err != nil {
					return err
				}

				//make a request to start a diagnosis job
				req, reqErr := http.NewRequest("POST", filepath.Join(mynahSettings.ApiPrefix, "ic/diagnosis/start"), bytes.NewBuffer(jsonBody))
				if reqErr != nil {
					return reqErr
				}
				req.Header.Add("Content-Type", "application/json")

				//handle user creation endpoint
				c.Router.HandleHTTPRequest("POST", "ic/diagnosis/start",
					startICDiagnosisJob(c.DBProvider, c.AsyncProvider, c.PyImplProvider, c.StorageProvider))

				//make the request
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("ic/diagnosis/start returned non-200: %v want %v", code, http.StatusOK)
					}

					//wait for the websocket response
					return <-errChan
				})
			})
		})
	})

	if err != nil {
		t.Fatalf("TestAPIStartDiagnosisJobEndpoint error: %s", err)
	}
}

//test the creation of an ic dataset
func TestICDatasetCreationEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {

			//handle user creation endpoint
			c.Router.HandleHTTPRequest("POST", "icdataset/create", icDatasetCreate(c.DBProvider))

			return c.WithCreateFile(user, "test_contents", func(file *model.MynahFile) error {
				//create the request
				reqContents := createICDatasetRequest{
					Name:  "test_dataset",
					Files: make(map[string]string),
				}

				//set the class for the file
				reqContents.Files[file.Uuid] = "class1"

				jsonBody, jsonErr := json.Marshal(reqContents)
				if jsonErr != nil {
					return jsonErr
				}

				req, reqErr := http.NewRequest("POST", filepath.Join(mynahSettings.ApiPrefix, "icdataset/create"), bytes.NewBuffer(jsonBody))
				if reqErr != nil {
					return reqErr
				}
				req.Header.Add("Content-Type", "application/json")

				//make the request
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("ic/diagnosis/start returned non-200: %v want %v", code, http.StatusOK)
					}

					//check that the user was inserted into the database
					var res createICDatasetResponse
					//check the body
					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
						return fmt.Errorf("failed to decode response %s", err)
					}

					//check for the user in the database (as a known admin)
					dbDataset, dbErr := c.DBProvider.GetICDataset(&res.Dataset.Uuid, user)
					if dbErr != nil {
						return fmt.Errorf("new user not found in database %s", dbErr)
					}

					//verify same
					if dbDataset.OrgId != user.OrgId {
						return fmt.Errorf("user from db (%v) not assigned same org id (%v)", dbDataset.OrgId, user.OrgId)
					}

					//check the files contents
					if fileData, found := dbDataset.Files[file.Uuid]; found {
						if fileData.CurrentClass != "class1" {
							return fmt.Errorf("file did not have expected class")
						}
					} else {
						return fmt.Errorf("file not included in new dataset")
					}

					return nil
				})
			})
		})
	})

	if err != nil {
		t.Fatalf("TestDatasetCreationEndpoint error: %s", err)
	}
}

//test the creation of an ic project
func TestICProjectCreationEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {

			//handle user creation endpoint
			c.Router.HandleHTTPRequest("POST", "icproject/create", icProjectCreate(c.DBProvider))

			return c.WithCreateICDataset(user, func(dataset *model.MynahICDataset) error {

				reqContents := createICProjectRequest{
					Name: "test_project",
					Datasets: []string{
						dataset.Uuid,
					},
				}

				jsonBody, jsonErr := json.Marshal(reqContents)
				if jsonErr != nil {
					return jsonErr
				}

				req, reqErr := http.NewRequest("POST", filepath.Join(mynahSettings.ApiPrefix, "icproject/create"), bytes.NewBuffer(jsonBody))
				if reqErr != nil {
					return reqErr
				}
				req.Header.Add("Content-Type", "application/json")

				//make the request
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("ic/diagnosis/start returned non-200: %v want %v", code, http.StatusOK)
					}

					//check that the user was inserted into the database
					var res createICProjectResponse
					//check the body
					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
						return fmt.Errorf("failed to decode response %s", err)
					}

					//check for the user in the database (as a known admin)
					dbProject, dbErr := c.DBProvider.GetICProject(&res.Project.Uuid, user)
					if dbErr != nil {
						return fmt.Errorf("new user not found in database %s", dbErr)
					}

					//verify same
					if dbProject.OrgId != user.OrgId {
						return fmt.Errorf("user from db (%v) not assigned same org id (%v)", dbProject.OrgId, user.OrgId)
					}

					if dbProject.Datasets[0] != dataset.Uuid {
						return fmt.Errorf("dataset id did not match")
					}

					return nil
				})
			})
		})
	})

	if err != nil {
		t.Fatalf("TestICProjectCreationEndpoint error: %s", err)
	}
}

//test upload and download endpoints
func TestAPIFileEndpoints(t *testing.T) {

	//TODO

	// mynahSettings := settings.DefaultSettings()
	//
	// //initialize auth
	// authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	// if authErr != nil {
	//   t.Errorf("failed to initialize auth %s", authErr)
	//   return
	// }
	//
	// //initialize the database connection
	// dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	// if dbErr != nil {
	//   t.Errorf("failed to initialize database connection %s", dbErr)
	//   return
	// }
	//
	// //initialize router
	// router := middleware.NewRouter(mynahSettings, authProvider, dbProvider)
	// //handle upload endpoint
	// router.HandleHTTPRequest("upload", handleFileUpload(settings, dbProvider, storageProvider))

}
