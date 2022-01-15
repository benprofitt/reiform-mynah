package db

import (
	"github.com/google/uuid"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"testing"
)

//Test basic database behavior
func TestBasicDBActions(t *testing.T) {
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

	//create a second user (should error)
	if createErr := dbProvider.CreateUser(&localUser, &admin); createErr == nil {
		t.Errorf("double user creation did not return error")
		return
	}

	//list users and verify same
	if userList, listErr := dbProvider.ListUsers(&admin); listErr == nil {
		//should only be one user
		if *userList[0] != localUser {
			t.Errorf("user in list (%v) not identical to local (%v)", *userList[0], localUser)
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
	if updateErr := dbProvider.UpdateUser(&localUser, &admin, "NameFirst", "NameLast"); updateErr == nil {
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
}
