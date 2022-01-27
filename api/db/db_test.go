// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"github.com/google/uuid"
	"reflect"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"testing"
)

//TODO run tests for both database providers by changing the settings and passing in

//Test basic database behavior
func TestBasicDBActionsUser(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	if authPErr != nil {
		t.Errorf("failed to create auth provider for test %s", authPErr)
		return
	}
	defer authProvider.Close()
	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	if dbPErr != nil {
		t.Errorf("failed to create database provider for test %s", dbPErr)
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
		t.Errorf("failed to create test user %s", createErr)
		return
	}

	if localUser.OrgId != admin.OrgId {
		t.Errorf("user did not inherit admin org id")
		return
	}

	//create a second user (should error)
	if createErr := dbProvider.CreateUser(&localUser, &admin); createErr == nil {
		t.Errorf("double user creation did not return error")
		return
	}

	//get user for auth by uuid
	if dbUser, getErr := dbProvider.GetUserForAuth(&localUser.Uuid); getErr == nil {
		//compare
		if *dbUser != localUser {
			t.Errorf("user from db (%v) not identical to local (%v)", *dbUser, localUser)
			return
		}
	} else {
		t.Errorf("failed to get user by uuid %s", getErr)
		return
	}

	//list users and verify same
	if userList, listErr := dbProvider.ListUsers(&admin); listErr == nil {
		//should only be one user
		if *userList[0] != localUser {
			t.Errorf("user in list (%v) not identical to local (%v)", *userList[0], localUser)
			return
		}
		if len(userList) > 1 {
			t.Errorf("more than one user in list (%d)", len(userList))
			return
		}
	} else {
		t.Errorf("failed to list users %s", listErr)
		return
	}

	//get the user and verify same
	if dbUser, getErr := dbProvider.GetUser(&localUser.Uuid, &admin); getErr == nil {
		//compare
		if *dbUser != localUser {
			t.Errorf("user from db (%v) not identical to local (%v)", *dbUser, localUser)
			return
		}
	} else {
		t.Errorf("failed to get user %s", getErr)
		return
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
				t.Errorf("user from db (%v) not identical to local (%v)", *dbUser, localUser)
				return
			}
		} else {
			t.Errorf("failed to get user (after update) %s", getErr)
			return
		}
	} else {
		t.Errorf("failed to update user %s", updateErr)
		return
	}

	//delete the user
	if deleteErr := dbProvider.DeleteUser(&localUser.Uuid, &admin); deleteErr == nil {
		//verify deleted
		if _, getErr := dbProvider.GetUser(&localUser.Uuid, &admin); getErr == nil {
			t.Errorf("failed to delete user from db")
			return
		}
	} else {
		t.Errorf("failed to delete user %s", deleteErr)
		return
	}
}

func TestBasicDBActionsProject(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	if authPErr != nil {
		t.Errorf("failed to create auth provider for test %s", authPErr)
	}
	defer authProvider.Close()
	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	if dbPErr != nil {
		t.Errorf("failed to create database provider for test %s", dbPErr)
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
		t.Errorf("failed to create test user %s", createErr)
		return
	}

	if project.UserPermissions[user.Uuid] != model.Owner {
		t.Errorf("user is not marked as project owner")
		return
	}

	if project.OrgId != user.OrgId {
		t.Errorf("project did not inherit user org id")
		return
	}

	//list projects and verify same
	if projectList, listErr := dbProvider.ListProjects(&user); listErr == nil {
		//should only be one project
		if !reflect.DeepEqual(*projectList[0], project) {
			t.Errorf("project in list (%v) not identical to local (%v)", *projectList[0], project)
			return
		}
		if len(projectList) > 1 {
			t.Errorf("more than one project in list (%d)", len(projectList))
			return
		}
	} else {
		t.Errorf("failed to list projects %s", listErr)
		return
	}

	//get the project and verify same
	if dbProject, getErr := dbProvider.GetProject(&project.Uuid, &user); getErr == nil {
		//compare
		if !reflect.DeepEqual(*dbProject, project) {
			t.Errorf("project from db (%v) not identical to local (%v)", *dbProject, project)
			return
		}
	} else {
		t.Errorf("failed to get project %s", getErr)
		return
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
				t.Errorf("project from db (%v) not identical to local (%v)", *dbProject, project)
				return
			}
		} else {
			t.Errorf("failed to get project (after update) %s", getErr)
			return
		}
	} else {
		t.Errorf("failed to update project %s", updateErr)
		return
	}

	//delete the project
	if deleteErr := dbProvider.DeleteProject(&project.Uuid, &user); deleteErr == nil {
		//verify deleted
		if _, getErr := dbProvider.GetProject(&project.Uuid, &user); getErr == nil {
			t.Errorf("failed to delete project from db")
			return
		}
	} else {
		t.Errorf("failed to delete project %s", deleteErr)
		return
	}
}
