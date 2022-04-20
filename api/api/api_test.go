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
	"path"
	"path/filepath"
	"reflect"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/test"
	"reiform.com/mynah/tools"
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
	req, reqErr := http.NewRequest("POST", path.Join(c.Settings.ApiPrefix, "admin", "user/create"), bytes.NewBuffer(jsonBody))
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
		var res AdminCreateUserResponse
		//check the body
		if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
			return fmt.Errorf("failed to decode response %s", err)
		}

		//check for the user in the database (as a known admin)
		dbUser, dbErr := c.DBProvider.GetUser(res.User.Uuid, user)
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
	mynahSettings.PythonSettings.ModulePath = "../../python"

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		filePath := fmt.Sprintf("file/{%s}/{%s}", fileKey, fileVersionIdKey)

		c.Router.HandleAdminRequest("POST", "user/create", adminCreateUser(c.DBProvider, c.AuthProvider))
		c.Router.HandleHTTPRequest("GET", filePath, handleViewFile(c.DBProvider, c.StorageProvider))

		testContents := "test contents"
		expectedType := "text/plain; charset=utf-8"

		//create a user
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			//create a file
			return c.WithCreateFile(user, testContents, func(file *model.MynahFile) error {
				//make a request for the file
				req, reqErr := http.NewRequest("GET", path.Join(mynahSettings.ApiPrefix, "file", string(file.Uuid), "latest"), nil)
				if reqErr != nil {
					return reqErr
				}

				//make a request for the file
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("get file returned non-200: %v want %v", code, http.StatusOK)
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
			expectedFileIds := tools.NewUniqueSet()
			expectedFileIds.Union("fileuuid1", "fileuuid2", "fileuuid3", "fileuuid4")
			//create a file
			return c.WithCreateFullICDataset(user, expectedFileIds.UuidVals(), func(dataset *model.MynahICDataset) error {

				errChan := make(chan error)
				readyChan := make(chan struct{})

				//listen for websocket response
				go c.WebsocketListener(jwt, 1, readyChan, errChan, func(res []byte) error {
					var report model.MynahICDiagnosisReport
					//parse as a report
					if err := json.Unmarshal(res, &report); err == nil {

						//generate expected image versionid
						expectedVersionId := model.MynahFileVersionId("6410687e280fef2ae3ed75a1c3a99ec7bc72d08f")

						reportFileIds := tools.NewUniqueSet()
						reportFileIds.UuidsUnion(report.ImageIds...)

						if !expectedFileIds.Equals(reportFileIds) {
							return fmt.Errorf("unexpected fileids set: %v vs %v", expectedFileIds.Vals(), reportFileIds.Vals())
						}

						expectedBreakdown := make(map[string]*model.MynahICDiagnosisReportBucket)
						expectedBreakdown["class1"] = &model.MynahICDiagnosisReportBucket{
							Bad:        0,
							Acceptable: 2,
						}
						expectedBreakdown["class2"] = &model.MynahICDiagnosisReportBucket{
							Bad:        2,
							Acceptable: 0,
						}

						if !reflect.DeepEqual(report.Breakdown, expectedBreakdown) {
							return fmt.Errorf("unexpected breakdown map: %v vs %v", report.Breakdown, expectedBreakdown)
						}

						expectedFileData1 := &model.MynahICDiagnosisReportImageMetadata{
							ImageVersionId: expectedVersionId,
							Class:          "class1",
							Mislabeled:     false,
							Point: model.MynahICDiagnosisReportPoint{
								X: 0,
								Y: 0,
							},
							OutlierSets: []string{},
						}

						if !reflect.DeepEqual(report.ImageData["fileuuid1"], expectedFileData1) {
							return fmt.Errorf("unexpected fileid1 data: %#v vs %#v", report.ImageData["fileuuid1"], expectedFileData1)
						}

						expectedFileData3 := &model.MynahICDiagnosisReportImageMetadata{
							ImageVersionId: expectedVersionId,
							Class:          "class2",
							Mislabeled:     true,
							Point: model.MynahICDiagnosisReportPoint{
								X: 0,
								Y: 0,
							},
							OutlierSets: []string{"mislabeled_images"},
						}

						if !reflect.DeepEqual(report.ImageData["fileuuid3"], expectedFileData3) {
							return fmt.Errorf("unexpected fileid3 data: %#v vs %#v", report.ImageData["fileuuid3"], expectedFileData3)
						}

						return nil
					} else {
						return err
					}
				})

				//wait for the websocket server to be up
				<-readyChan

				//create the request body
				reqBody := StartDiagnosisJobRequest{
					DatasetUuid: dataset.Uuid,
				}

				jsonBody, err := json.Marshal(reqBody)
				if err != nil {
					return err
				}

				//make a request to start a diagnosis job
				req, reqErr := http.NewRequest("POST", path.Join(mynahSettings.ApiPrefix, "dataset/ic/diagnosis/start"), bytes.NewBuffer(jsonBody))
				if reqErr != nil {
					return reqErr
				}
				req.Header.Add("Content-Type", "application/json")

				//handle user creation endpoint
				c.Router.HandleHTTPRequest("POST", "dataset/ic/diagnosis/start",
					startICDiagnosisJob(c.DBProvider, c.AsyncProvider, c.PyImplProvider, c.StorageProvider))

				//make the request
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("dataset/ic/diagnosis/start returned non-200: %v want %v", code, http.StatusOK)
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
			c.Router.HandleHTTPRequest("POST", "dataset/ic/create", icDatasetCreate(c.DBProvider, c.StorageProvider))

			return c.WithCreateFile(user, "test_contents", func(file *model.MynahFile) error {
				//create the request
				reqContents := CreateICDatasetRequest{
					Name:  "test_dataset",
					Files: make(map[model.MynahUuid]string),
				}

				//set the class for the file
				reqContents.Files[file.Uuid] = "class1"

				jsonBody, jsonErr := json.Marshal(reqContents)
				if jsonErr != nil {
					return jsonErr
				}

				req, reqErr := http.NewRequest("POST", filepath.Join(mynahSettings.ApiPrefix, "dataset/ic/create"), bytes.NewBuffer(jsonBody))
				if reqErr != nil {
					return reqErr
				}
				req.Header.Add("Content-Type", "application/json")

				//make the request
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("dataset/ic/diagnosis/start returned non-200: %v want %v", code, http.StatusOK)
					}

					//check that the user was inserted into the database
					var res model.MynahICDataset
					//check the body
					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
						return fmt.Errorf("failed to decode response %s", err)
					}

					//check for dataset in database (as a known admin)
					dbDataset, dbErr := c.DBProvider.GetICDataset(res.Uuid, user)
					if dbErr != nil {
						return fmt.Errorf("new dataset not found in database %s", dbErr)
					}

					//verify same
					if dbDataset.OrgId != user.OrgId {
						return fmt.Errorf("dataset from db (%v) not assigned same org id (%v)", dbDataset.OrgId, user.OrgId)
					}

					//check the files contents
					if fileData, found := dbDataset.Versions["0"].Files[file.Uuid]; found {
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

func TestAPIReportFilter(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			return c.WithCreateICDiagnosisReport(user, func(report *model.MynahICDiagnosisReport) error {

				c.Router.HandleHTTPRequest("GET",
					fmt.Sprintf("dataset/ic/report/{%s}", icReportKey),
					icDiagnosisReportView(c.DBProvider))

				//make a standard request
				requestPath := path.Join(mynahSettings.ApiPrefix, "dataset/ic/report", string(report.Uuid))
				req, reqErr := http.NewRequest("GET", requestPath, nil)
				if reqErr != nil {
					return reqErr
				}

				if err := c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					var res model.MynahICDiagnosisReport
					if err := json.NewDecoder(rr.Body).Decode(&res); err == nil {
						if len(res.ImageData) != 0 || len(res.ImageIds) != 0 || len(res.Breakdown) != 0 {
							return errors.New("unexpected data in report")
						}
						return nil
					} else {
						return err
					}

				}); err != nil {
					return err
				}

				//add class filter
				requestPath = fmt.Sprintf("%s%s", path.Join(mynahSettings.ApiPrefix, "dataset/ic/report", string(report.Uuid)), "?class=class1")
				req, reqErr = http.NewRequest("GET", requestPath, nil)
				if reqErr != nil {
					return reqErr
				}

				if err := c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					var res model.MynahICDiagnosisReport
					if err := json.NewDecoder(rr.Body).Decode(&res); err == nil {
						if len(res.ImageData) != 1 || len(res.ImageIds) != 1 || len(res.Breakdown) != 1 {
							return errors.New("unexpected data in report")
						}
						return nil
					} else {
						return err
					}

				}); err != nil {
					return err
				}

				//add bad images filter
				requestPath = fmt.Sprintf("%s%s", path.Join(mynahSettings.ApiPrefix, "dataset/ic/report", string(report.Uuid)), "?class=class1&class=class2&bad_images=true")
				req, reqErr = http.NewRequest("GET", requestPath, nil)
				if reqErr != nil {
					return reqErr
				}

				if err := c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					var res model.MynahICDiagnosisReport
					if err := json.NewDecoder(rr.Body).Decode(&res); err == nil {
						if len(res.ImageData) != 1 || len(res.ImageIds) != 1 || len(res.Breakdown) != 2 {
							return errors.New("unexpected data in report")
						}
						return nil
					} else {
						return err
					}

				}); err != nil {
					return err
				}

				return nil
			})
		})
	})

	if err != nil {
		t.Fatalf("TestAPIReportFilter error: %s", err)
	}
}

//test dataset list
func TestListDatasetsEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			return c.WithCreateICDataset(user, func(icDataset *model.MynahICDataset) error {
				return c.WithCreateODDataset(user, func(odDataset *model.MynahODDataset) error {
					c.Router.HandleHTTPRequest("GET", "dataset/list",
						allDatasetList(c.DBProvider))

					req, reqErr := http.NewRequest("GET", path.Join(mynahSettings.ApiPrefix, "dataset/list"), nil)
					if reqErr != nil {
						return reqErr
					}

					return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
						if code != http.StatusOK {
							return fmt.Errorf("dataset/list returned non 200 status: %d", code)
						}

						//TODO decode result, check dataset types

						return nil
					})
				})
			})
		})
	})

	if err != nil {
		t.Fatalf("TestListDatasetsEndpoint error: %s", err)
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
