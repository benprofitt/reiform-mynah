// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"github.com/google/uuid"
	"os"
	"reflect"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"testing"
	"time"
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

//TODO run tests for both database providers by changing the settings and passing in

//Test basic database behavior
func TestBasicDBActionsUser(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	if authPErr != nil {
		t.Fatalf("failed to create auth provider for test %s", authPErr)
		return
	}
	defer authProvider.Close()
	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	if dbPErr != nil {
		t.Fatalf("failed to create database provider for test %s", dbPErr)
		return
	}
	defer dbProvider.Close()

	admin := model.MynahUser{
		IsAdmin: true,
		OrgId:   uuid.New().String(),
	}

	//create a user
	localUser := model.MynahUser{
		Uuid:      uuid.New().String(),
		NameFirst: "test_user_first",
		NameLast:  "test_user_last",
		IsAdmin:   false,
	}

	//create the user in the database
	if createErr := dbProvider.CreateUser(&localUser, &admin); createErr != nil {
		t.Fatalf("failed to create test user %s", createErr)
	}

	if localUser.OrgId != admin.OrgId {
		t.Fatalf("user did not inherit admin org id")
	}

	//create a second user (should error)
	if createErr := dbProvider.CreateUser(&localUser, &admin); createErr == nil {
		t.Fatalf("double user creation did not return error")
	}

	//get user for auth by uuid
	if dbUser, getErr := dbProvider.GetUserForAuth(&localUser.Uuid); getErr == nil {
		//compare
		if *dbUser != localUser {
			t.Fatalf("user from db (%v) not identical to local (%v)", *dbUser, localUser)
		}
	} else {
		t.Fatalf("failed to get user by uuid %s", getErr)
	}

	//list users and verify same
	if userList, listErr := dbProvider.ListUsers(&admin); listErr == nil {
		//should only be one user
		if *userList[0] != localUser {
			t.Fatalf("user in list (%v) not identical to local (%v)", *userList[0], localUser)
		}
		if len(userList) > 1 {
			t.Fatalf("more than one user in list (%d)", len(userList))
		}
	} else {
		t.Fatalf("failed to list users %s", listErr)
	}

	//get the user and verify same
	if dbUser, getErr := dbProvider.GetUser(&localUser.Uuid, &admin); getErr == nil {
		//compare
		if *dbUser != localUser {
			t.Fatalf("user from db (%v) not identical to local (%v)", *dbUser, localUser)
		}
	} else {
		t.Fatalf("failed to get user %s", getErr)
	}

	//update some fields
	localUser.NameFirst = "new_name_first"
	localUser.NameLast = "new_name_last"

	//update the user
	if updateErr := dbProvider.UpdateUser(&localUser, &admin, "name_first", "name_last"); updateErr == nil {
		//get the user and verify same
		if dbUser, getErr := dbProvider.GetUser(&localUser.Uuid, &admin); getErr == nil {
			//look through list
			if *dbUser != localUser {
				t.Fatalf("user from db (%v) not identical to local (%v)", *dbUser, localUser)
			}
		} else {
			t.Fatalf("failed to get user (after update) %s", getErr)
		}
	} else {
		t.Fatalf("failed to update user %s", updateErr)
	}

	//delete the user
	if deleteErr := dbProvider.DeleteUser(&localUser.Uuid, &admin); deleteErr == nil {
		//verify deleted
		if _, getErr := dbProvider.GetUser(&localUser.Uuid, &admin); getErr == nil {
			t.Fatalf("failed to delete user from db")
		}
	} else {
		t.Fatalf("failed to delete user %s", deleteErr)
	}
}

func TestBasicDBActionsProject(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	if authPErr != nil {
		t.Fatalf("failed to create auth provider for test %s", authPErr)
	}
	defer authProvider.Close()
	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	if dbPErr != nil {
		t.Fatalf("failed to create database provider for test %s", dbPErr)
	}
	defer dbProvider.Close()

	//create a user
	user := model.MynahUser{
		Uuid:    uuid.New().String(),
		OrgId:   uuid.New().String(),
		IsAdmin: false,
	}

	project := model.MynahProject{
		UserPermissions: make(map[string]model.ProjectPermissions),
		ProjectName:     "project_test",
	}

	//create the project in the database
	if createErr := dbProvider.CreateProject(&project, &user); createErr != nil {
		t.Fatalf("failed to create test project %s", createErr)
	}

	if project.UserPermissions[user.Uuid] != model.Owner {
		t.Fatalf("user is not marked as project owner")
	}

	if project.OrgId != user.OrgId {
		t.Fatalf("project did not inherit user org id")
	}

	//list projects and verify same
	if projectList, listErr := dbProvider.ListProjects(&user); listErr == nil {
		//should only be one project
		if !reflect.DeepEqual(*projectList[0], project) {
			t.Fatalf("project in list (%v) not identical to local (%v)", *projectList[0], project)
		}
		if len(projectList) > 1 {
			t.Fatalf("more than one project in list (%d)", len(projectList))
		}
	} else {
		t.Fatalf("failed to list projects %s", listErr)
	}

	//get the project and verify same
	if dbProject, getErr := dbProvider.GetProject(&project.Uuid, &user); getErr == nil {
		//compare
		if !reflect.DeepEqual(*dbProject, project) {
			t.Fatalf("project from db (%v) not identical to local (%v)", *dbProject, project)
		}
	} else {
		t.Fatalf("failed to get project %s", getErr)
	}

	//update some fields
	project.UserPermissions["new_user_uuid"] = model.Read
	project.ProjectName = "new_project_name"

	//update the project
	if updateErr := dbProvider.UpdateProject(&project, &user, "project_name", "user_permissions"); updateErr == nil {
		//get the project and verify same
		if dbProject, getErr := dbProvider.GetProject(&project.Uuid, &user); getErr == nil {
			//compare
			if !reflect.DeepEqual(*dbProject, project) {
				t.Fatalf("project from db (%v) not identical to local (%v)", *dbProject, project)
			}
		} else {
			t.Fatalf("failed to get project (after update) %s", getErr)
		}
	} else {
		t.Fatalf("failed to update project %s", updateErr)
	}

	//delete the project
	if deleteErr := dbProvider.DeleteProject(&project.Uuid, &user); deleteErr == nil {
		//verify deleted
		if _, getErr := dbProvider.GetProject(&project.Uuid, &user); getErr == nil {
			t.Fatalf("failed to delete project from db")
		}
	} else {
		t.Fatalf("failed to delete project %s", deleteErr)
	}
}

func TestBasicDBActionsDataset(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	if authPErr != nil {
		t.Fatalf("failed to create auth provider for test %s", authPErr)
	}
	defer authProvider.Close()
	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	if dbPErr != nil {
		t.Fatalf("failed to create database provider for test %s", dbPErr)
	}
	defer dbProvider.Close()

	//create a user
	user := model.MynahUser{
		Uuid:    uuid.New().String(),
		OrgId:   uuid.New().String(),
		IsAdmin: false,
	}

	//create a normal dataset
	dataset := model.MynahDataset{
		DatasetName: "dataset_test",
	}

	icDataset := model.MynahICDataset{
		model.MynahDataset{
			DatasetName: "ic_dataset_test",
		},
		make([]string, 0),
	}

	//create the projects in the database
	if createErr := dbProvider.CreateDataset(&dataset, &user); createErr != nil {
		t.Fatalf("failed to create test dataset %s", createErr)
	}

	if createErr := dbProvider.CreateICDataset(&icDataset, &user); createErr != nil {
		t.Fatalf("failed to create test ic dataset %s", createErr)
	}

	if (dataset.OrgId != user.OrgId) || (icDataset.OrgId != user.OrgId) {
		t.Fatalf("project did not inherit user org id")
	}

	//list datasets and verify same
	if datasetList, listErr := dbProvider.ListDatasets(&user); listErr == nil {
		//should only be one project
		if !reflect.DeepEqual(*datasetList[0], dataset) {
			t.Fatalf("dataset in list (%v) not identical to local (%v)", *datasetList[0], dataset)
		}
		if len(datasetList) > 1 {
			t.Fatalf("more than one dataset in list (%d)", len(datasetList))
		}
	} else {
		t.Fatalf("failed to list datasets %s", listErr)
	}

	//list datasets and verify same
	if datasetList, listErr := dbProvider.ListICDatasets(&user); listErr == nil {
		//should only be one project
		if !reflect.DeepEqual(*datasetList[0], icDataset) {
			t.Fatalf("dataset in list (%v) not identical to local (%v)", *datasetList[0], icDataset)
		}
		if len(datasetList) > 1 {
			t.Fatalf("more than one ic dataset in list (%d)", len(datasetList))
		}
	} else {
		t.Fatalf("failed to list ic datasets %s", listErr)
	}

	//get the datasets and verify same
	if dbDataset, getErr := dbProvider.GetDataset(&dataset.Uuid, &user); getErr == nil {
		//compare
		if !reflect.DeepEqual(*dbDataset, dataset) {
			t.Fatalf("dataset from db (%v) not identical to local (%v)", *dbDataset, dataset)
		}
	} else {
		t.Fatalf("failed to get dataset %s", getErr)
	}

	if dbDataset, getErr := dbProvider.GetICDataset(&icDataset.Uuid, &user); getErr == nil {
		//compare
		if !reflect.DeepEqual(*dbDataset, icDataset) {
			t.Fatalf("ic dataset from db (%v) not identical to local (%v)", *dbDataset, icDataset)
		}
	} else {
		t.Fatalf("failed to get ic dataset %s", getErr)
	}

	//update some fields
	dataset.DatasetName = "new_dataset_name"
	icDataset.DatasetName = "new_icdataset_name"

	//update the dataset
	if updateErr := dbProvider.UpdateDataset(&dataset, &user, "dataset_name"); updateErr == nil {
		//get the dataset and verify same
		if dbDataset, getErr := dbProvider.GetDataset(&dataset.Uuid, &user); getErr == nil {
			//compare
			if !reflect.DeepEqual(*dbDataset, dataset) {
				t.Fatalf("dataset from db (%v) not identical to local (%v)", *dbDataset, dataset)
			}
		} else {
			t.Fatalf("failed to get dataset (after update) %s", getErr)
		}
	} else {
		t.Fatalf("failed to update dataset %s", updateErr)
	}

	if updateErr := dbProvider.UpdateICDataset(&icDataset, &user, "dataset_name"); updateErr == nil {
		//get the dataset and verify same
		if dbDataset, getErr := dbProvider.GetICDataset(&icDataset.Uuid, &user); getErr == nil {
			//compare
			if !reflect.DeepEqual(*dbDataset, icDataset) {
				t.Fatalf("ic dataset from db (%v) not identical to local (%v)", *dbDataset, icDataset)
			}
		} else {
			t.Fatalf("failed to get ic dataset (after update) %s", getErr)
		}
	} else {
		t.Fatalf("failed to update ic dataset %s", updateErr)
	}

	//delete the dataset
	if deleteErr := dbProvider.DeleteDataset(&dataset.Uuid, &user); deleteErr == nil {
		//verify deleted
		if _, getErr := dbProvider.GetDataset(&dataset.Uuid, &user); getErr == nil {
			t.Fatalf("failed to delete dataset from db")
		}
	} else {
		t.Fatalf("failed to delete dataset %s", deleteErr)
	}

	if deleteErr := dbProvider.DeleteICDataset(&icDataset.Uuid, &user); deleteErr == nil {
		//verify deleted
		if _, getErr := dbProvider.GetICDataset(&icDataset.Uuid, &user); getErr == nil {
			t.Fatalf("failed to delete dataset from db")
		}
	} else {
		t.Fatalf("failed to delete dataset %s", deleteErr)
	}
}

func TestBasicDBFileUpdate(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	if authPErr != nil {
		t.Fatalf("failed to create auth provider for test %s", authPErr)
	}
	defer authProvider.Close()
	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	if dbPErr != nil {
		t.Fatalf("failed to create database provider for test %s", dbPErr)
	}
	defer dbProvider.Close()

	//create a user
	user := model.MynahUser{
		Uuid:    uuid.New().String(),
		OrgId:   uuid.New().String(),
		IsAdmin: false,
	}

	var file model.MynahFile

	//create the project in the database
	if createErr := dbProvider.CreateFile(&file, &user); createErr != nil {
		t.Fatalf("failed to create test file %s", createErr)
	}

	//check the modification time
	modTime := file.LastModified

	//wait
	time.Sleep(2 * time.Second)

	//update the file (no fields)
	if updateErr := dbProvider.UpdateFile(&file, &user); updateErr != nil {
		t.Fatalf("failed to update test file %s", updateErr)
	}

	//read the file
	if dbFile, getErr := dbProvider.GetFile(&file.Uuid, &user); getErr == nil {
		//check updated time
		if dbFile.LastModified <= modTime {
			t.Fatalf("last_modified not updated %d <= %d", dbFile.LastModified, modTime)
		}
	} else {
		t.Fatalf("failed to get ic dataset %s", getErr)
	}

	//update file and explicitly pass modification key
	if updateErr := dbProvider.UpdateFile(&file, &user, "last_modified"); updateErr != nil {
		t.Fatalf("failed to update test file %s", updateErr)
	}
}
