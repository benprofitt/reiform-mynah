// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"reiform.com/mynah/containers"
	"reiform.com/mynah/db"
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
	mynahSettings.PythonSettings.PythonExecutable = "../../python/mynah.py"

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		//handle user creation endpoint
		c.Router.HandleAdminRequest("POST", "user/create", adminCreateUser(c.DBProvider, c.AuthProvider))

		//create as admin
		err := c.WithCreateUser(true, func(user *model.MynahUser, jwt string) error {
			return makeCreateUserReq(user, jwt, c)
		})
		require.NoError(t, err)

		//create as non-admin
		err = c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			return makeCreateUserReq(user, jwt, c)
		})

		require.NotNil(t, err, "create user as non admin did not produce error as expected")
		return nil
	})
	require.NoErrorf(t, err, "Should not cause error")
}

func TestFileGetEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.PythonExecutable = "../../python/mynah.py"

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
				require.NoError(t, reqErr)

				//make a request for the file
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					require.Equal(t, http.StatusOK, code, "Get file should return 200 status")
					require.Equal(t, int64(len(testContents)), rr.Result().ContentLength, "File contents length not expected")
					require.Equal(t, expectedType, rr.Result().Header.Get("Content-Type"), "Content type not expected")
					return nil
				})
			})
		})
	})
	require.NoErrorf(t, err, "Should not cause error")
}

func TestAPIStartDiagnosisJobEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.PythonExecutable = "../../python/diagnosis_data_test.py"

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		//create a user
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			startingFileids := containers.NewUniqueSet[model.MynahUuid]()
			startingFileids.Union("fileuuid1", "fileuuid2", "fileuuid3", "fileuuid4")
			//create a file
			return c.WithCreateICDataset(user, startingFileids.Vals(), func(dataset *model.MynahICDataset) error {
				//create the request body
				reqBody := ICProcessJobRequest{
					Tasks:       []model.MynahICProcessTaskType{model.ICProcessDiagnoseMislabeledImagesTask},
					DatasetUuid: dataset.Uuid,
				}

				jsonBody, err := json.Marshal(reqBody)
				require.NoError(t, err)

				//make a request to start a diagnosis job
				req, reqErr := http.NewRequest("POST", path.Join(mynahSettings.ApiPrefix, "dataset/ic/process/start"), bytes.NewBuffer(jsonBody))
				require.NoError(t, reqErr)

				req.Header.Add("Content-Type", "application/json")

				//handle user creation endpoint
				c.Router.HandleHTTPRequest("POST", "dataset/ic/process/start",
					icProcessJob(c.DBProvider, c.AsyncProvider, c.ImplProvider, c.StorageProvider))
				c.Router.HandleHTTPRequest("GET",
					fmt.Sprintf("data/json/{%s}", idKey),
					getDataJSON(c.DBProvider))

				//make the request
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					require.Equal(t, http.StatusOK, code, "Request should return 200 status")

					//get the response
					var res ICProcessJobResponse
					//check the body
					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
						return fmt.Errorf("failed to decode response %s", err)
					}

					//wait for task completion
					return c.AsyncTaskWaiter(user, res.TaskUuid, func() error {
						//get the dataset with the updated report section
						updatedDataset, err := c.DBProvider.GetICDataset(dataset.Uuid, user, db.NewMynahDBColumns())
						require.NoError(t, err)

						//get the report
						requestPath := path.Join(mynahSettings.ApiPrefix, fmt.Sprintf("data/json/%s", string(updatedDataset.Reports["1"].DataId)))
						req, reqErr := http.NewRequest("GET", requestPath, nil)
						require.NoError(t, reqErr)

						return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
							var actual model.MynahICDatasetReport
							if err := json.NewDecoder(rr.Body).Decode(&actual); err != nil {
								return err
							}

							expected := model.MynahICDatasetReport{
								Points: map[model.MynahClassName][]*model.MynahICDatasetReportPoint{
									"class1": {
										{
											FileId:         "fileuuid1",
											ImageVersionId: "6410687e280fef2ae3ed75a1c3a99ec7bc72d08f",
											X:              0,
											Y:              0,
											OriginalClass:  "class1",
										},
										{
											FileId:         "fileuuid4",
											ImageVersionId: "6410687e280fef2ae3ed75a1c3a99ec7bc72d08f",
											X:              0,
											Y:              0,
											OriginalClass:  "class1",
										},
									},
									"class2": {
										{
											FileId:         "fileuuid3",
											ImageVersionId: "6410687e280fef2ae3ed75a1c3a99ec7bc72d08f",
											X:              0,
											Y:              0,
											OriginalClass:  "class1",
										},
										{
											FileId:         "fileuuid2",
											ImageVersionId: "6410687e280fef2ae3ed75a1c3a99ec7bc72d08f",
											X:              0,
											Y:              0,
											OriginalClass:  "class1",
										},
									},
								},
								Tasks: []*model.MynahICProcessTaskReportData{
									{
										Type: "ic::diagnose::mislabeled_images",
										Metadata: &model.MynahICProcessTaskDiagnoseMislabeledImagesReport{
											ClassLabelErrors: map[model.MynahClassName]*model.MynahICProcessTaskDiagnoseMislabeledImagesReportClass{
												"class1": {
													Mislabeled: []model.MynahUuid{
														"fileuuid1",
													},
													Correct: []model.MynahUuid{
														"fileuuid4",
													},
												},
												"class2": {
													Mislabeled: []model.MynahUuid{
														"fileuuid3",
													},
													Correct: []model.MynahUuid{
														"fileuuid2",
													},
												},
											},
										},
									},
								},
							}

							for className, class := range expected.Points {
								require.Contains(t, actual.Points, className)
								require.ElementsMatch(t, class, actual.Points[className])
							}

							for className, _ := range actual.Points {
								require.Contains(t, expected.Points, className)
							}

							require.ElementsMatch(t, expected.Tasks, actual.Tasks)

							//check that the dataset was updated correctly
							dbDataset, err := c.DBProvider.GetICDataset(dataset.Uuid, user, db.NewMynahDBColumns())
							require.NoError(t, err)

							require.Equal(t, model.MynahDatasetVersionId("1"), dbDataset.LatestVersion, "Unexpected dataset version")
							require.Equal(t, 2, len(dbDataset.Versions), "Unexpected dataset version count")
							return nil
						})
					})
				})
			})
		})
	})
	require.NoErrorf(t, err, "Should not cause error")
}

//test the creation of an ic dataset
func TestICDatasetCreationEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.PythonExecutable = "../../python/mynah.py"

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {

			//handle user creation endpoint
			c.Router.HandleHTTPRequest("POST", "dataset/ic/create", icDatasetCreate(c.DBProvider))

			return c.WithCreateFile(user, "test_contents", func(file *model.MynahFile) error {
				//create the request
				reqContents := CreateICDatasetRequest{
					Name:  "test_dataset",
					Files: make(map[model.MynahUuid]model.MynahClassName),
				}

				//set the class for the file
				reqContents.Files[file.Uuid] = "class1"

				jsonBody, jsonErr := json.Marshal(reqContents)
				require.NoError(t, jsonErr)

				req, reqErr := http.NewRequest("POST", filepath.Join(mynahSettings.ApiPrefix, "dataset/ic/create"), bytes.NewBuffer(jsonBody))
				require.NoError(t, reqErr)

				req.Header.Add("Content-Type", "application/json")

				//make the request
				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					require.Equal(t, http.StatusOK, code, "Request should return 200 status")

					//check that the user was inserted into the database
					var res model.MynahICDataset
					//check the body
					require.NoError(t, json.NewDecoder(rr.Body).Decode(&res))

					//check for dataset in database (as a known admin)
					dbDataset, dbErr := c.DBProvider.GetICDataset(res.Uuid, user, db.NewMynahDBColumns())
					require.NoError(t, dbErr)

					//verify same
					require.Equal(t, user.OrgId, dbDataset.OrgId, "Org ids should be the same")

					//check the files contents
					fileData, found := dbDataset.Versions["0"].Files[file.Uuid]
					require.True(t, found)
					require.Equal(t, model.MynahClassName("class1"), fileData.CurrentClass)
					return nil
				})
			})
		})
	})

	require.NoErrorf(t, err, "Should not cause error")
}

func TestAPIReportFilter(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.PythonExecutable = "../../python/mynah.py"

	//load the testing concd /vag	text
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			expectedFileIds := containers.NewUniqueSet[model.MynahUuid]()
			expectedFileIds.Union("fileuuid1", "fileuuid2", "fileuuid3", "fileuuid4")

			return c.WithCreateICDataset(user, expectedFileIds.Vals(), func(dataset *model.MynahICDataset) error {

				c.Router.HandleHTTPRequest("GET",
					fmt.Sprintf("data/json/{%s}", idKey),
					getDataJSON(c.DBProvider))

				//make a standard request
				requestPath := path.Join(mynahSettings.ApiPrefix, fmt.Sprintf("data/json/%s", string(dataset.Reports["0"].DataId)))
				req, reqErr := http.NewRequest("GET", requestPath, nil)
				require.NoError(t, reqErr)

				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					var res model.MynahICDatasetReport
					require.NoError(t, json.NewDecoder(rr.Body).Decode(&res))
					// TODO check result
					return nil
				})
			})
		})
	})
	require.NoErrorf(t, err, "Should not cause error")
}

//test dataset list
func TestListDatasetsEndpoint(t *testing.T) {
	mynahSettings := settings.DefaultSettings()
	mynahSettings.PythonSettings.PythonExecutable = "../../python/mynah.py"

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
			return c.WithCreateICDataset(user, []model.MynahUuid{}, func(icDataset *model.MynahICDataset) error {
				return c.WithCreateODDataset(user, func(odDataset *model.MynahODDataset) error {
					c.Router.HandleHTTPRequest("GET", "dataset/list",
						allDatasetList(c.DBProvider))

					req, reqErr := http.NewRequest("GET", path.Join(mynahSettings.ApiPrefix, "dataset/list"), nil)
					require.NoError(t, reqErr)

					return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
						require.Equal(t, http.StatusOK, code, "Request should return 200 status")

						//TODO decode result, check dataset types

						return nil
					})
				})
			})
		})
	})
	require.NoErrorf(t, err, "Should not cause error")
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

//test size detection
func TestImageSizeDetection(t *testing.T) {
	s := settings.DefaultSettings()
	s.PythonSettings.PythonExecutable = "../../python/image_metadata_data_test.py"

	err := test.WithTestContext(s, func(c *test.TestContext) error {
		jpegPath := "../../docs/test_image.jpg"
		pngPath := "../../docs/mynah_arch_1-13-21.drawio.png"
		notImagePath := "../../docs/ipc.md"

		user := model.MynahUser{
			Uuid: "owner",
		}

		err := c.WithCreateFileFromPath(&user, jpegPath, func(file *model.MynahFile) error {
			return c.StorageProvider.GetStoredFile(file, model.OriginalVersionId, func(path string) error {
				require.NoError(t, c.ImplProvider.ImageMetadata(&user, path, file, file.Versions[model.OriginalVersionId]))

				width := file.Versions[model.OriginalVersionId].Metadata.GetDefaultInt(model.MetadataWidth, 0)
				height := file.Versions[model.OriginalVersionId].Metadata.GetDefaultInt(model.MetadataHeight, 0)

				require.Equal(t, int64(4032), width, "Unexpected width")
				require.Equal(t, int64(3024), height, "Unexpected height")

				return nil
			})
		})

		require.NoError(t, err)

		err = c.WithCreateFileFromPath(&user, pngPath, func(file *model.MynahFile) error {
			return c.StorageProvider.GetStoredFile(file, model.OriginalVersionId, func(path string) error {
				require.NoError(t, c.ImplProvider.ImageMetadata(&user, path, file, file.Versions[model.OriginalVersionId]))

				width := file.Versions[model.OriginalVersionId].Metadata.GetDefaultInt(model.MetadataWidth, 0)
				height := file.Versions[model.OriginalVersionId].Metadata.GetDefaultInt(model.MetadataHeight, 0)

				require.Equal(t, int64(3429), width, "Unexpected width")
				require.Equal(t, int64(2316), height, "Unexpected height")

				return nil
			})
		})

		require.NoError(t, err)

		err = c.WithCreateFileFromPath(&user, notImagePath, func(file *model.MynahFile) error {
			return c.StorageProvider.GetStoredFile(file, model.OriginalVersionId, func(path string) error {
				return c.ImplProvider.ImageMetadata(&user, path, file, file.Versions[model.OriginalVersionId])
			})
		})

		require.Errorf(t, err, "Non image should cause error")
		return nil
	})
	require.NoErrorf(t, err, "Should not cause error")
}
