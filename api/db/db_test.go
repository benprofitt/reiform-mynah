// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"github.com/stretchr/testify/require"
	"os"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
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

//TODO run tests for both database providers by changing the settings and passing in

//Test basic database behavior
func TestBasicDBActionsUser(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	require.NoError(t, authPErr)
	defer authProvider.Close()

	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	require.NoError(t, dbPErr)
	defer dbProvider.Close()

	admin := model.MynahUser{
		IsAdmin: true,
		OrgId:   model.NewMynahUuid(),
	}

	localUser, err := dbProvider.CreateUser(&admin, func(user *model.MynahUser) error {
		user.NameFirst = "test_user_first"
		user.NameLast = "test_user_last"
		user.IsAdmin = false
		return nil
	})
	require.NoError(t, err)

	//create the user in the database
	require.NoError(t, err)

	require.Equal(t, admin.OrgId, localUser.OrgId)

	//try to create a user with the same uuid
	_, err = dbProvider.CreateUser(&admin, func(user *model.MynahUser) error {
		user.Uuid = localUser.Uuid
		return nil
	})

	//create a second user (should error)
	require.Errorf(t, err, "Double user creation should return error")

	//get user for auth by uuid
	dbUser, getErr := dbProvider.GetUserForAuth(localUser.Uuid)
	require.NoError(t, getErr)
	require.Equal(t, *localUser, *dbUser)

	userList, listErr := dbProvider.ListUsers(&admin)
	require.NoError(t, listErr)
	require.Equal(t, *localUser, *userList[0])
	require.Equal(t, len(userList), 1)

	dbUser, getErr = dbProvider.GetUser(localUser.Uuid, &admin)
	require.NoError(t, getErr)
	require.Equal(t, *localUser, *dbUser)

	//update some fields
	localUser.NameFirst = "new_name_first"
	localUser.NameLast = "new_name_last"

	require.NoError(t, dbProvider.UpdateUser(localUser, &admin, NewMynahDBColumns(model.NameLastCol, model.NameFirstCol)))

	dbUser, getErr = dbProvider.GetUser(localUser.Uuid, &admin)
	require.NoError(t, getErr)
	require.Equal(t, *localUser, *dbUser)

	require.NoError(t, dbProvider.DeleteUser(localUser.Uuid, &admin))

	_, getErr = dbProvider.GetUser(localUser.Uuid, &admin)
	require.Error(t, getErr)
}

func TestBasicDBActionsICDataset(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	require.NoError(t, authPErr)
	defer authProvider.Close()

	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	require.NoError(t, dbPErr)
	defer dbProvider.Close()

	//create a user
	user := model.MynahUser{
		Uuid:    model.NewMynahUuid(),
		OrgId:   model.NewMynahUuid(),
		IsAdmin: false,
	}

	icDataset, err := dbProvider.CreateICDataset(&user, func(icDataset *model.MynahICDataset) error {
		icDataset.DatasetName = "ic_dataset_test"
		icDataset.Versions["0"] = &model.MynahICDatasetVersion{}
		return nil
	})

	require.NoError(t, err)

	require.NoError(t, err)
	require.Equal(t, user.OrgId, icDataset.OrgId)

	omitDataset, err := dbProvider.GetICDataset(icDataset.Uuid, &user, NewMynahDBColumnsNoDeps(model.VersionsColName))
	require.NoError(t, err)
	require.Nil(t, omitDataset.Versions)

	datasetList, listErr := dbProvider.ListICDatasets(&user)
	require.NoError(t, listErr)
	require.Equal(t, *datasetList[0], *icDataset)
	require.Equal(t, len(datasetList), 1)

	dbDataset, getErr := dbProvider.GetICDataset(icDataset.Uuid, &user, NewMynahDBColumns())
	require.NoError(t, getErr)
	require.Equal(t, *dbDataset, *icDataset)

	//update some fields
	icDataset.DatasetName = "new_icdataset_name"

	require.NoError(t, dbProvider.UpdateICDataset(icDataset, &user, NewMynahDBColumns(model.DatasetNameCol)))
	dbDataset, getErr = dbProvider.GetICDataset(icDataset.Uuid, &user, NewMynahDBColumns())
	require.NoError(t, getErr)
	require.Equal(t, *dbDataset, *icDataset)

	require.Error(t, dbProvider.UpdateICDataset(icDataset, &user, NewMynahDBColumns("uuid")))

	require.NoError(t, dbProvider.DeleteICDataset(icDataset.Uuid, &user))

	_, getErr = dbProvider.GetICDataset(icDataset.Uuid, &user, NewMynahDBColumns())
	require.Error(t, getErr)
}

func TestBasicDBActionsFile(t *testing.T) {
	s := settings.DefaultSettings()
	authProvider, authPErr := auth.NewAuthProvider(s)
	require.NoError(t, authPErr)
	defer authProvider.Close()

	dbProvider, dbPErr := NewDBProvider(s, authProvider)
	require.NoError(t, dbPErr)
	defer dbProvider.Close()

	//create a user
	user := model.MynahUser{
		Uuid:    model.NewMynahUuid(),
		OrgId:   model.NewMynahUuid(),
		IsAdmin: false,
	}

	rangeRequest := 10

	//add more files and then make range request
	var uuids []model.MynahUuid

	for i := 0; i < rangeRequest; i++ {

		_, err := dbProvider.CreateFile(&user, func(file *model.MynahFile) error {
			uuids = append(uuids, file.Uuid)
			return nil
		})

		require.NoError(t, err)
	}

	//make a range request
	dbFiles, rangeErr := dbProvider.GetFiles(uuids, &user)
	require.NoError(t, rangeErr)
	require.Equal(t, len(dbFiles), rangeRequest)

	//make a request for one with the range operator
	dbFile, rangeErr := dbProvider.GetFiles([]model.MynahUuid{uuids[0]}, &user)
	require.NoError(t, rangeErr)
	require.Equal(t, 1, len(dbFile))

	//make an empty request
	noFiles, rangeErr := dbProvider.GetFiles([]model.MynahUuid{}, &user)
	require.NoError(t, rangeErr)
	require.Equal(t, 0, len(noFiles))
}
