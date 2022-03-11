// Copyright (c) 2022 by Reiform. All Rights Reserved.

package graphql

import (
	"encoding/json"
	"fmt"
	"github.com/google/uuid"
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

//the response expected from gql
type expectedUserResponse struct {
	Data struct {
		User model.MynahUser `json:"user"`
	} `json:"data"`
}

//the response type expected for listing users
type expectedListUserResponse struct {
	Data struct {
		List []model.MynahUser `json:"list"`
	} `json:"data"`
}

type expectedUpdateUserResponse struct {
	Data struct {
		Update model.MynahUser `json:"update"`
	} `json:"data"`
}

//the response expected from gql
type expectedICDatasetResponse struct {
	Data struct {
		ICDataset model.MynahICDataset `json:"icdataset"`
	} `json:"data"`
}

//the response type expected for listing users
type expectedListICDatasetResponse struct {
	Data struct {
		List []model.MynahICDataset `json:"list"`
	} `json:"data"`
}

type expectedUpdateICDatasetResponse struct {
	Data struct {
		Update model.MynahICDataset `json:"update"`
	} `json:"data"`
}

//the response expected from gql
type expectedICProjectResponse struct {
	Data struct {
		ICProject model.MynahICProject `json:"icproject"`
	} `json:"data"`
}

//the response type expected for listing users
type expectedListICProjectResponse struct {
	Data struct {
		List []model.MynahICProject `json:"list"`
	} `json:"data"`
}

type expectedUpdateICProjectResponse struct {
	Data struct {
		Update model.MynahICProject `json:"update"`
	} `json:"data"`
}

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

//Test user query (by uuid and list)
func TestQueryUser(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		//create gql user query resolver
		userHandler, userErr := UserQueryResolver(c.DBProvider)
		if userErr != nil {
			return userErr
		}
		//handle user list endpoint
		c.Router.HandleHTTPRequest("GET", "graphql/user", userHandler)

		//create a different user
		return c.WithCreateUser(false, func(nonAdmin *model.MynahUser, _ string) error {

			return c.WithCreateUser(true, func(admin *model.MynahUser, jwt string) error {
				reqString := fmt.Sprintf("graphql/user?query={user(uuid:\"%s\"){uuid,name_first,name_last}}", nonAdmin.Uuid)
				//make a request
				req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
				if reqErr != nil {
					return reqErr
				}

				err := c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("get user returned non-200: %v want %v", code, http.StatusOK)
					}

					var res expectedUserResponse
					//check the body
					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
						return fmt.Errorf("failed to decode response %s", err)
					}

					if res.Data.User.Uuid != nonAdmin.Uuid {
						return fmt.Errorf("unexpected uuid: %s", res.Data.User.Uuid)
					}

					if res.Data.User.NameFirst != "first" {
						return fmt.Errorf("unexpected first name: %s", res.Data.User.NameFirst)
					}

					return nil
				})

				if err != nil {
					return err
				}

				//make a request
				req, reqErr = http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix,
					"graphql/user?query={list{uuid,name_last,name_first}}"), nil)
				if reqErr != nil {
					return reqErr
				}

				err = c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("list user returned non-200: %v want %v", code, http.StatusOK)
					}

					var res expectedListUserResponse
					//check the body
					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
						return fmt.Errorf("failed to decode response %s", err)
					}

					if len(res.Data.List) != 2 {
						return fmt.Errorf("unexpected user list length: %d", len(res.Data.List))
					}

					return nil
				})

				return err
			})
		})
	})

	if err != nil {
		t.Fatalf("TestQueryUser error: %s", err)
	}
}

//Test user update
func TestUpdateUser(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		//create gql user query resolver
		userHandler, userErr := UserQueryResolver(c.DBProvider)
		if userErr != nil {
			return userErr
		}
		//handle user update endpoint
		c.Router.HandleHTTPRequest("GET", "graphql/user", userHandler)

		//create a different user
		return c.WithCreateUser(false, func(nonAdmin *model.MynahUser, _ string) error {

			return c.WithCreateUser(true, func(admin *model.MynahUser, jwt string) error {
				newFirst := uuid.NewString()
				newLast := uuid.NewString()
				reqString := fmt.Sprintf(
					"graphql/user?query=mutation+_{update(uuid:\"%s\",name_first:\"%s\",name_last:\"%s\"){uuid,name_first,name_last}}",
					nonAdmin.Uuid,
					newFirst,
					newLast)
				//make a request
				req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
				if reqErr != nil {
					return reqErr
				}

				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
					//check the result
					if code != http.StatusOK {
						return fmt.Errorf("update user returned non-200: %v want %v", code, http.StatusOK)
					}

					var res expectedUpdateUserResponse
					//check the body
					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
						return fmt.Errorf("failed to decode response %s", err)
					}

					if res.Data.Update.Uuid != nonAdmin.Uuid {
						return fmt.Errorf("unexpected uuid: %s", res.Data.Update.Uuid)
					}

					if res.Data.Update.NameFirst != newFirst {
						return fmt.Errorf("unexpected first name: %s", res.Data.Update.NameFirst)
					}

					//request the user from the db
					if user, err := c.DBProvider.GetUser(&nonAdmin.Uuid, admin); err == nil {
						if (user.NameFirst != newFirst) || (user.NameLast != newLast) {
							return fmt.Errorf("unexpected updated name combination: (%s != %s) and (%s != %s)",
								user.NameFirst, newFirst, user.NameLast, newLast)
						}

					} else {
						return err
					}

					return nil
				})
			})
		})

	})

	if err != nil {
		t.Fatalf("TestUpdateUser error: %s", err)
	}
}

//Test user deletion
func TestDeleteUser(t *testing.T) {
	mynahSettings := settings.DefaultSettings()

	//load the testing context
	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
		//create gql user query resolver
		userHandler, userErr := UserQueryResolver(c.DBProvider)
		if userErr != nil {
			return userErr
		}
		//handle user deletion endpoint
		c.Router.HandleHTTPRequest("GET", "graphql/user", userHandler)

		return c.WithCreateUser(true, func(admin *model.MynahUser, jwt string) error {

			//create a user to delete
			nonAdmin, err := c.DBProvider.CreateUser(admin, func(user *model.MynahUser) {})
			if err != nil {
				return err
			}

			reqString := fmt.Sprintf(
				"graphql/user?query=mutation+_{delete(uuid:\"%s\"){uuid}}",
				nonAdmin.Uuid)
			//make a request
			req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
			if reqErr != nil {
				return reqErr
			}

			return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
				//check the result
				if code != http.StatusOK {
					return fmt.Errorf("delete user returned non-200: %v want %v", code, http.StatusOK)
				}

				//request the deleted user from the db
				if _, err := c.DBProvider.GetUser(&nonAdmin.Uuid, admin); err == nil {
					return fmt.Errorf("expected error when requesting deleted user")
				}

				return nil
			})
		})

	})

	if err != nil {
		t.Fatalf("TestDeleteUser error: %s", err)
	}
}

////Test ic dataset query (by uuid and list)
//func TestQueryICDataset(t *testing.T) {
//	mynahSettings := settings.DefaultSettings()
//
//	//load the testing context
//	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
//		//create gql user query resolver
//		datasetHandler, datasetErr := ICDatasetQueryResolver(c.DBProvider)
//		if datasetErr != nil {
//			return datasetErr
//		}
//		//handle dataset list
//		c.Router.HandleHTTPRequest("graphql/icdataset", datasetHandler)
//
//		//create the dataset owner
//		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
//
//			//create a dataset
//			return c.WithCreateICDataset(user, func(dataset *model.MynahICDataset) error {
//
//				reqString := fmt.Sprintf("graphql/icdataset?query={icdataset(uuid:\"%s\"){uuid,dataset_name,classes}}", dataset.Uuid)
//				//make a request
//				req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
//				if reqErr != nil {
//					return reqErr
//				}
//
//				err := c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
//					//check the result
//					if code != http.StatusOK {
//						return fmt.Errorf("get dataset returned non-200: %v want %v", code, http.StatusOK)
//					}
//
//					fmt.Printf("%s\n", rr.Body.String())
//
//					var res expectedICDatasetResponse
//					//check the body
//					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
//						return fmt.Errorf("failed to decode response %s", err)
//					}
//
//					if res.Data.ICDataset.Uuid != dataset.Uuid {
//						return fmt.Errorf("unexpected uuid: %s", res.Data.ICDataset.Uuid)
//					}
//
//					if res.Data.ICDataset.DatasetName != dataset.DatasetName {
//						return fmt.Errorf("unexpected name: %s", res.Data.ICDataset.DatasetName)
//					}
//
//					if !reflect.DeepEqual(res.Data.ICDataset.Classes, dataset.Classes) {
//						return fmt.Errorf("classes didn't match: %v vs. %v", res.Data.ICDataset.Classes, dataset.Classes)
//					}
//
//					return nil
//				})
//
//				if err != nil {
//					return err
//				}
//
//				//make a request
//				req, reqErr = http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix,
//					"graphql/icdataset?query={list{uuid,dataset_name,classes}}"), nil)
//				if reqErr != nil {
//					return reqErr
//				}
//
//				err = c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
//					//check the result
//					if code != http.StatusOK {
//						return fmt.Errorf("list datasets returned non-200: %v want %v", code, http.StatusOK)
//					}
//
//					var res expectedListICDatasetResponse
//					//check the body
//					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
//						return fmt.Errorf("failed to decode response %s", err)
//					}
//
//					if len(res.Data.List) != 1 {
//						return fmt.Errorf("unexpected dataset list length: %d", len(res.Data.List))
//					}
//
//					return nil
//				})
//
//				return err
//			})
//		})
//	})
//
//	if err != nil {
//		t.Fatalf("TestQueryICDataset error: %s", err)
//	}
//}
//
////Test dataset update
//func TestUpdateICDataset(t *testing.T) {
//	mynahSettings := settings.DefaultSettings()
//
//	//load the testing context
//	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
//		datasetHandler, datasetErr := ICDatasetQueryResolver(c.DBProvider)
//		if datasetErr != nil {
//			return datasetErr
//		}
//		//handle dataset update
//		c.Router.HandleHTTPRequest("graphql/icdataset", datasetHandler)
//
//		//create a different user
//		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
//
//			return c.WithCreateICDataset(user, func(dataset *model.MynahICDataset) error {
//				newName := uuid.NewString()
//				reqString := fmt.Sprintf(
//					"graphql/icdataset?query=mutation+_{update(uuid:\"%s\",dataset_name:\"%s\",classes:[\"%s\"]){uuid,dataset_name,classes}}",
//					dataset.Uuid,
//					newName,
//					"test_class")
//				//make a request
//				req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
//				if reqErr != nil {
//					return reqErr
//				}
//
//				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
//					//check the result
//					if code != http.StatusOK {
//						return fmt.Errorf("update dataset returned non-200: %v want %v", code, http.StatusOK)
//					}
//
//					var res expectedUpdateICDatasetResponse
//					//check the body
//					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
//						return fmt.Errorf("failed to decode response %s", err)
//					}
//
//					if res.Data.Update.Uuid != dataset.Uuid {
//						return fmt.Errorf("unexpected uuid: %s", res.Data.Update.Uuid)
//					}
//
//					if res.Data.Update.DatasetName != newName {
//						return fmt.Errorf("unexpected name: %s", res.Data.Update.DatasetName)
//					}
//
//					//request the dataset from the db
//					if dbDataset, err := c.DBProvider.GetICDataset(&dataset.Uuid, user); err == nil {
//						if (dbDataset.DatasetName != newName) || !reflect.DeepEqual(dbDataset.Classes, dataset.Classes) {
//							return fmt.Errorf("dataset returned and dataset in db aren't the same")
//						}
//
//					} else {
//						return err
//					}
//
//					return nil
//				})
//			})
//		})
//
//	})
//
//	if err != nil {
//		t.Fatalf("TestUpdateICDataset error: %s", err)
//	}
//}
//
////Test dataset deletion
//func TestDeleteICDataset(t *testing.T) {
//	mynahSettings := settings.DefaultSettings()
//
//	//load the testing context
//	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
//		datasetHandler, datasetErr := ICDatasetQueryResolver(c.DBProvider)
//		if datasetErr != nil {
//			return datasetErr
//		}
//		//handle dataset delete
//		c.Router.HandleHTTPRequest("graphql/icdataset", datasetHandler)
//
//		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
//
//			dataset, err := c.DBProvider.CreateICDataset(user, func(*model.MynahICDataset) {})
//			if err != nil {
//				return fmt.Errorf("failed to create dataset in database: %s", err)
//			}
//
//			reqString := fmt.Sprintf(
//				"graphql/icdataset?query=mutation+_{delete(uuid:\"%s\"){uuid}}",
//				dataset.Uuid)
//			//make a request
//			req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
//			if reqErr != nil {
//				return reqErr
//			}
//
//			return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
//				//check the result
//				if code != http.StatusOK {
//					return fmt.Errorf("delete user returned non-200: %v want %v", code, http.StatusOK)
//				}
//
//				//request the deleted user from the db
//				if _, err := c.DBProvider.GetICDataset(&dataset.Uuid, user); err == nil {
//					return fmt.Errorf("expected error when requesting deleted dataset")
//				}
//
//				return nil
//			})
//		})
//
//	})
//
//	if err != nil {
//		t.Fatalf("TestDeleteICDataset error: %s", err)
//	}
//}
//
////Test ic project query (by uuid and list)
//func TestQueryICProject(t *testing.T) {
//	mynahSettings := settings.DefaultSettings()
//
//	//load the testing context
//	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
//		//create gql project query resolver
//		projectHandler, projectErr := ICProjectQueryResolver(c.DBProvider)
//		if projectErr != nil {
//			return projectErr
//		}
//		//handle dataset list
//		c.Router.HandleHTTPRequest("graphql/icproject", projectHandler)
//
//		//create the dataset owner
//		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
//
//			//create a project
//			return c.WithCreateICProject(user, func(project *model.MynahICProject) error {
//
//				reqString := fmt.Sprintf("graphql/icproject?query={icproject(uuid:\"%s\"){uuid,project_name}}", project.Uuid)
//				//make a request
//				req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
//				if reqErr != nil {
//					return reqErr
//				}
//
//				err := c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
//					//check the result
//					if code != http.StatusOK {
//						return fmt.Errorf("get project returned non-200: %v want %v", code, http.StatusOK)
//					}
//
//					var res expectedICProjectResponse
//					//check the body
//					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
//						return fmt.Errorf("failed to decode response %s", err)
//					}
//
//					if res.Data.ICProject.Uuid != project.Uuid {
//						return fmt.Errorf("unexpected uuid: %s", res.Data.ICProject.Uuid)
//					}
//
//					if res.Data.ICProject.ProjectName != project.ProjectName {
//						return fmt.Errorf("unexpected name: %s", res.Data.ICProject.ProjectName)
//					}
//					return nil
//				})
//
//				if err != nil {
//					return err
//				}
//
//				//make a request
//				req, reqErr = http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix,
//					"graphql/icproject?query={list{uuid,project_name}}"), nil)
//				if reqErr != nil {
//					return reqErr
//				}
//
//				err = c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
//					//check the result
//					if code != http.StatusOK {
//						return fmt.Errorf("list projects returned non-200: %v want %v", code, http.StatusOK)
//					}
//
//					var res expectedListICProjectResponse
//					//check the body
//					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
//						return fmt.Errorf("failed to decode response %s", err)
//					}
//
//					if len(res.Data.List) != 1 {
//						return fmt.Errorf("unexpected project list length: %d", len(res.Data.List))
//					}
//
//					return nil
//				})
//
//				return err
//			})
//		})
//	})
//
//	if err != nil {
//		t.Fatalf("TestQueryICProject error: %s", err)
//	}
//}
//
////Test project update
//func TestUpdateICProject(t *testing.T) {
//	mynahSettings := settings.DefaultSettings()
//
//	//load the testing context
//	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
//		//create gql project query resolver
//		projectHandler, projectErr := ICProjectQueryResolver(c.DBProvider)
//		if projectErr != nil {
//			return projectErr
//		}
//		//handle dataset list
//		c.Router.HandleHTTPRequest("graphql/icproject", projectHandler)
//
//		//create a different user
//		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
//
//			return c.WithCreateICProject(user, func(project *model.MynahICProject) error {
//				newName := uuid.NewString()
//				reqString := fmt.Sprintf(
//					"graphql/icproject?query=mutation+_{update(uuid:\"%s\",project_name:\"%s\"){uuid,project_name}}",
//					project.Uuid,
//					newName)
//				//make a request
//				req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
//				if reqErr != nil {
//					return reqErr
//				}
//
//				return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
//					//check the result
//					if code != http.StatusOK {
//						return fmt.Errorf("update project returned non-200: %v want %v", code, http.StatusOK)
//					}
//
//					var res expectedUpdateICProjectResponse
//					//check the body
//					if err := json.NewDecoder(rr.Body).Decode(&res); err != nil {
//						return fmt.Errorf("failed to decode response %s", err)
//					}
//
//					if res.Data.Update.Uuid != project.Uuid {
//						return fmt.Errorf("unexpected uuid: %s", res.Data.Update.Uuid)
//					}
//
//					if res.Data.Update.ProjectName != newName {
//						return fmt.Errorf("unexpected name: %s", res.Data.Update.ProjectName)
//					}
//
//					//request the dataset from the db
//					if dbProject, err := c.DBProvider.GetICProject(&project.Uuid, user); err == nil {
//						if dbProject.ProjectName != newName {
//							return fmt.Errorf("project returned and project in db aren't the same")
//						}
//
//					} else {
//						return err
//					}
//
//					return nil
//				})
//			})
//		})
//
//	})
//
//	if err != nil {
//		t.Fatalf("TestUpdateICProject error: %s", err)
//	}
//}
//
////Test project deletion
//func TestDeleteICProject(t *testing.T) {
//	mynahSettings := settings.DefaultSettings()
//
//	//load the testing context
//	err := test.WithTestContext(mynahSettings, func(c *test.TestContext) error {
//		//create gql project query resolver
//		projectHandler, projectErr := ICProjectQueryResolver(c.DBProvider)
//		if projectErr != nil {
//			return projectErr
//		}
//		//handle dataset list
//		c.Router.HandleHTTPRequest("graphql/icproject", projectHandler)
//
//		return c.WithCreateUser(false, func(user *model.MynahUser, jwt string) error {
//
//			project, err := c.DBProvider.CreateProject(user, func(*model.MynahProject) {})
//			if err != nil {
//				return fmt.Errorf("failed to create project in database: %s", err)
//			}
//
//			reqString := fmt.Sprintf(
//				"graphql/icproject?query=mutation+_{delete(uuid:\"%s\"){uuid}}",
//				project.Uuid)
//			//make a request
//			req, reqErr := http.NewRequest("GET", filepath.Join(mynahSettings.ApiPrefix, reqString), nil)
//			if reqErr != nil {
//				return reqErr
//			}
//
//			return c.WithHTTPRequest(req, jwt, func(code int, rr *httptest.ResponseRecorder) error {
//				//check the result
//				if code != http.StatusOK {
//					return fmt.Errorf("delete project returned non-200: %v want %v", code, http.StatusOK)
//				}
//
//				//request the deleted user from the db
//				if _, err := c.DBProvider.GetICProject(&project.Uuid, user); err == nil {
//					return fmt.Errorf("expected error when requesting deleted project")
//				}
//
//				return nil
//			})
//		})
//
//	})
//
//	if err != nil {
//		t.Fatalf("TestDeleteICProject error: %s", err)
//	}
//}
