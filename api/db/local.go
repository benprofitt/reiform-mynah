// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"errors"
	"fmt"
	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"
	"os"
	"path/filepath"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"xorm.io/xorm"
)

//local database client implements DBProvider
type localDB struct {
	//the XORM engine
	engine *xorm.Engine
}

//create the database file if it doesn't exist return
func checkDBFile(path string) (exists bool, err error) {
	if _, err := os.Stat(path); err == nil {
		return true, nil
	} else if errors.Is(err, os.ErrNotExist) {
		if file, err := os.Create(filepath.Clean(path)); err == nil {
			return false, file.Close()
		} else {
			return false, err
		}
	} else {
		return false, fmt.Errorf("failed to identify whether database already exists: %s", err)
	}
}

//create a new organization in the database and a starting admin user
func (d *localDB) createLocalOrg(authProvider auth.AuthProvider) error {

	tempAdmin := model.MynahUser{
		OrgId:   uuid.NewString(),
		IsAdmin: true,
	}

	//create an admin
	admin, err := d.CreateUser(&tempAdmin, func(user *model.MynahUser) {
		user.IsAdmin = true
	})

	if err != nil {
		return err
	}

	//create the initial admin user, organization id
	jwt, err := authProvider.GetUserAuth(admin)
	if err != nil {
		return err
	}

	//log the initial information
	log.Infof("created organization %s", admin.OrgId)
	log.Infof("created initial admin JWT for org (%s): %s", admin.OrgId, jwt)

	return nil
}

//create a new local db instance
func newLocalDB(mynahSettings *settings.MynahSettings, authProvider auth.AuthProvider) (*localDB, error) {
	//check if the database file has been created
	exists, createErr := checkDBFile(mynahSettings.DBSettings.LocalPath)
	if createErr != nil {
		return nil, createErr
	}

	//create the gorm engine
	engine, engineErr := xorm.NewEngine("sqlite3", mynahSettings.DBSettings.LocalPath)
	if engineErr != nil {
		return nil, engineErr
	}

	db := localDB{
		engine: engine,
	}

	if !exists {
		//create tables and initial orgs
		tableErr := engine.CreateTables(&model.MynahUser{},
			&model.MynahICProject{},
			&model.MynahProject{},
			&model.MynahFile{},
			&model.MynahICDataset{},
			&model.MynahDataset{})

		if tableErr != nil {
			return nil, tableErr
		}

		syncErr := engine.Sync2(&model.MynahUser{},
			&model.MynahICProject{},
			&model.MynahProject{},
			&model.MynahFile{},
			&model.MynahICDataset{},
			&model.MynahDataset{})

		if syncErr != nil {
			return nil, syncErr
		}

		//create initial organization structure
		for i := 0; i < mynahSettings.DBSettings.InitialOrgCount; i++ {
			if err := db.createLocalOrg(authProvider); err != nil {
				return nil, err
			}
		}

		log.Warnf("created local database %s", mynahSettings.DBSettings.LocalPath)
	}

	return &db, nil
}

// GetUserForAuth Get a user by uuid or return an error
func (d *localDB) GetUserForAuth(uuid *string) (*model.MynahUser, error) {
	user := model.MynahUser{
		Uuid: *uuid,
	}

	found, err := d.engine.Get(&user)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("user %s not found", *uuid)
	}
	return &user, nil
}

// GetUser Get a user other than self (must be admin)
func (d *localDB) GetUser(uuid *string, requestor *model.MynahUser) (*model.MynahUser, error) {
	if user, err := d.GetUserForAuth(uuid); err == nil {
		//verify that this user has permission
		if commonErr := commonGetUser(user, requestor); commonErr != nil {
			return nil, commonErr
		}
		return user, nil
	} else {
		return nil, err
	}
}

// GetProject get a project by id or return an error
func (d *localDB) GetProject(uuid *string, requestor *model.MynahUser) (*model.MynahProject, error) {
	project := model.MynahProject{
		Uuid: *uuid,
	}

	found, err := d.engine.Where("org_id = ?", requestor.OrgId).Get(&project)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("project %s not found", *uuid)
	}

	//check that the user has permission
	if commonErr := commonGetProject(&project, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &project, nil
}

// GetICProject get a project by id or return an error, second arg is requestor
func (d *localDB) GetICProject(uuid *string, requestor *model.MynahUser) (*model.MynahICProject, error) {
	project := model.MynahICProject{
		model.MynahProject{
			Uuid: *uuid,
		},
		make([]string, 0),
	}

	found, err := d.engine.Where("org_id = ?", requestor.OrgId).Get(&project)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("project %s not found", *uuid)
	}

	//check that the user has permission
	if commonErr := commonGetProject(&project, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &project, nil
}

// GetFile get a file from the database
func (d *localDB) GetFile(uuid *string, requestor *model.MynahUser) (*model.MynahFile, error) {
	file := model.MynahFile{
		Uuid: *uuid,
	}

	found, err := d.engine.Where("org_id = ?", requestor.OrgId).Get(&file)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("file %s not found", *uuid)
	}

	//check that the user has permission
	if commonErr := commonGetFile(&file, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &file, nil
}

// GetFiles get multiple files by id
func (d *localDB) GetFiles(uuids []string, requestor *model.MynahUser) (res []*model.MynahFile, err error) {
	var files []*model.MynahFile

	//request a set of uuids within the org
	if err = d.engine.Where("org_id = ?", requestor.OrgId).In("uuid", uuids).Find(&files); err != nil {
		return nil, err
	}

	for _, f := range files {
		//check that the user has permission
		if commonErr := commonGetFile(f, requestor); commonErr == nil {
			//add to the filtered list
			res = append(res, f)
		} else {
			log.Warnf("user %s failed to view file %s", requestor.Uuid, f.Uuid)
		}
	}

	return res, nil
}

// GetDataset get a dataset from the database
func (d *localDB) GetDataset(uuid *string, requestor *model.MynahUser) (*model.MynahDataset, error) {
	dataset := model.MynahDataset{
		Uuid: *uuid,
	}

	found, err := d.engine.Where("org_id = ?", requestor.OrgId).Get(&dataset)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("dataset %s not found", *uuid)
	}

	//check that the user has permission
	if commonErr := commonGetDataset(&dataset, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &dataset, nil
}

// GetICDataset get a dataset from the database
func (d *localDB) GetICDataset(uuid *string, requestor *model.MynahUser) (*model.MynahICDataset, error) {
	dataset := model.MynahICDataset{
		model.MynahDataset{
			Uuid: *uuid,
		},
		make(map[string]model.MynahICDatasetFile),
	}

	found, err := d.engine.Where("org_id = ?", requestor.OrgId).Get(&dataset)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("dataset %s not found", *uuid)
	}

	//check that the user has permission
	if commonErr := commonGetDataset(&dataset, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &dataset, nil
}

// GetICDatasets get multiple ic datasets from the database
func (d *localDB) GetICDatasets(uuids []string, requestor *model.MynahUser) (res []*model.MynahICDataset, err error) {
	var datasets []*model.MynahICDataset

	//request a set of uuids within the org
	if err = d.engine.Where("org_id = ?", requestor.OrgId).In("uuid", uuids).Find(&datasets); err != nil {
		return nil, err
	}

	for _, d := range datasets {
		//check that the user has permission
		if commonErr := commonGetDataset(d, requestor); commonErr == nil {
			//add to the filtered list
			res = append(res, d)
		} else {
			log.Warnf("user %s failed to view ic dataset %s", requestor.Uuid, d.Uuid)
		}
	}

	return res, nil
}

// ListUsers list all users
func (d *localDB) ListUsers(requestor *model.MynahUser) (users []*model.MynahUser, err error) {
	if commonErr := commonListUsers(requestor); commonErr != nil {
		return users, commonErr
	}

	//list users
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&users)
	return users, err
}

// ListProjects list all projects
func (d *localDB) ListProjects(requestor *model.MynahUser) (projects []*model.MynahProject, err error) {
	//list projects
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&projects)
	//filter for the projects that this user can view
	return commonListProjects(projects, requestor), err
}

// ListICProjects list all projects, arg is requestor
func (d *localDB) ListICProjects(requestor *model.MynahUser) (projects []*model.MynahICProject, err error) {
	//list projects
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&projects)
	//filter for the projects that this user can view
	return commonListICProjects(projects, requestor), err
}

// ListFiles list all files, arg is requestor
func (d *localDB) ListFiles(requestor *model.MynahUser) (files []*model.MynahFile, err error) {
	//list files
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&files)
	//filter for the files that this user can view
	return commonListFiles(files, requestor), err
}

// ListDatasets list all datasets, arg is requestor
func (d *localDB) ListDatasets(requestor *model.MynahUser) (datasets []*model.MynahDataset, err error) {
	//list datasets
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&datasets)
	//filter for the datasets that this user can view
	return commonListDatasets(datasets, requestor), err
}

// ListICDatasets list all datasets, arg is requestor
func (d *localDB) ListICDatasets(requestor *model.MynahUser) (datasets []*model.MynahICDataset, err error) {
	//list datasets
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&datasets)
	//filter for the datasets that this user can view
	return commonListICDatasets(datasets, requestor), err
}

// CreateUser create a new user
func (d *localDB) CreateUser(creator *model.MynahUser, precommit func(*model.MynahUser)) (*model.MynahUser, error) {
	user, err := commonCreateUser(creator)

	if err != nil {
		return nil, err
	}

	//call handler to make changes before commit
	precommit(user)

	affected, err := d.engine.Insert(user)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("user %s not created (no records affected)", user.Uuid)
	}
	return user, nil
}

// CreateProject create a new project
func (d *localDB) CreateProject(creator *model.MynahUser, precommit func(*model.MynahProject)) (*model.MynahProject, error) {
	project := commonCreateProject(creator)
	//call handler to make changes before commit
	precommit(project)

	affected, err := d.engine.Insert(project)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("project %s not created (no records affected)", project.Uuid)
	}
	return project, nil
}

// CreateICProject create a new project, second arg is creator
func (d *localDB) CreateICProject(creator *model.MynahUser, precommit func(*model.MynahICProject)) (*model.MynahICProject, error) {
	project := commonCreateICProject(creator)

	precommit(project)

	affected, err := d.engine.Insert(project)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("project %s not created (no records affected)", project.Uuid)
	}
	return project, nil
}

// CreateFile create a new file, second arg is creator
func (d *localDB) CreateFile(creator *model.MynahUser, precommit func(*model.MynahFile) error) (*model.MynahFile, error) {
	file := commonCreateFile(creator)

	//since we can't update files once created, we may need to fail during creation when writing to local storage
	if err := precommit(file); err != nil {
		return nil, err
	}

	affected, err := d.engine.Insert(file)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("file %s not created (no records affected)", file.Uuid)
	}
	return file, nil
}

// CreateDataset create a new dataset
func (d *localDB) CreateDataset(creator *model.MynahUser, precommit func(*model.MynahDataset)) (*model.MynahDataset, error) {
	dataset := commonCreateDataset(creator)

	precommit(dataset)

	affected, err := d.engine.Insert(dataset)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("dataset %s not created (no records affected)", dataset.Uuid)
	}
	return dataset, nil
}

// CreateICDataset create a new dataset
func (d *localDB) CreateICDataset(creator *model.MynahUser, precommit func(*model.MynahICDataset)) (*model.MynahICDataset, error) {
	dataset := commonCreateICDataset(creator)

	precommit(dataset)

	affected, err := d.engine.Insert(dataset)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("dataset %s not created (no records affected)", dataset.Uuid)
	}
	return dataset, nil
}

// UpdateUser update a user in the database
func (d *localDB) UpdateUser(user *model.MynahUser, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateUser(user, requestor, keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(user)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("user %s not updated (no records affected)", user.Uuid)
	}
	return nil
}

// UpdateProject update a project in the database
func (d *localDB) UpdateProject(project *model.MynahProject, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateProject(project, requestor, keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(project)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("project %s not updated (no records affected)", project.Uuid)
	}
	return nil
}

// UpdateICProject update a project in the database
func (d *localDB) UpdateICProject(project *model.MynahICProject, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateProject(project, requestor, keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(project)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("project %s not updated (no records affected)", project.Uuid)
	}
	return nil
}

// UpdateDataset update a dataset
func (d *localDB) UpdateDataset(dataset *model.MynahDataset, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateDataset(dataset, requestor, keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("dataset %s not updated (no records affected)", dataset.Uuid)
	}
	return nil
}

// UpdateICDataset update a dataset
func (d *localDB) UpdateICDataset(dataset *model.MynahICDataset, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateDataset(dataset, requestor, keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("dataset %s not updated (no records affected)", dataset.Uuid)
	}
	return nil
}

// DeleteUser delete a user in the database
func (d *localDB) DeleteUser(uuid *string, requestor *model.MynahUser) error {
	if commonErr := commonDeleteUser(uuid, requestor); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Delete(&model.MynahUser{Uuid: *uuid})
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("user %s not deleted (no records affected)", *uuid)
	}
	return nil
}

// DeleteProject delete a project in the database
func (d *localDB) DeleteProject(uuid *string, requestor *model.MynahUser) error {
	project, getErr := d.GetProject(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the project to check permissions
	if commonErr := commonDeleteProject(project, requestor); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Delete(project)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("project %s not deleted (no records affected)", *uuid)
	}
	return nil
}

// DeleteICProject delete a project in the database, second arg is requestor
func (d *localDB) DeleteICProject(uuid *string, requestor *model.MynahUser) error {
	project, getErr := d.GetICProject(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the project to check permissions
	if commonErr := commonDeleteProject(project, requestor); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Delete(project)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("project %s not deleted (no records affected)", *uuid)
	}
	return nil
}

// DeleteFile delete a file in the database, second arg is requestor
func (d *localDB) DeleteFile(uuid *string, requestor *model.MynahUser) error {
	file, getErr := d.GetFile(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the project to check permissions
	if commonErr := commonDeleteFile(file, requestor); commonErr != nil {
		return commonErr
	}

	affected, err := d.engine.Delete(file)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("file %s not deleted (no records affected)", *uuid)
	}
	return nil
}

// DeleteDataset delete a dataset
func (d *localDB) DeleteDataset(uuid *string, requestor *model.MynahUser) error {
	dataset, getErr := d.GetDataset(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the project to check permissions
	if commonErr := commonDeleteDataset(dataset, requestor); commonErr != nil {
		return commonErr
	}

	affected, err := d.engine.Delete(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("dataset %s not deleted (no records affected)", *uuid)
	}
	return nil
}

// DeleteICDataset delete a dataset
func (d *localDB) DeleteICDataset(uuid *string, requestor *model.MynahUser) error {
	dataset, getErr := d.GetICDataset(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the project to check permissions
	if commonErr := commonDeleteDataset(dataset, requestor); commonErr != nil {
		return commonErr
	}

	affected, err := d.engine.Delete(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("dataset %s not deleted (no records affected)", *uuid)
	}
	return nil
}

// Close close the client connection on shutdown
func (d *localDB) Close() {
	log.Infof("local database engine shutdown")
	err := d.engine.Close()
	if err != nil {
		log.Warnf("error closing database engine: %s", err)
	}
}
