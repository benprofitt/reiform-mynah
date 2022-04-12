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
	admin, err := d.CreateUser(&tempAdmin, func(user *model.MynahUser) error {
		user.IsAdmin = true
		return nil
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
			&model.MynahFile{},
			&model.MynahICDataset{},
			&model.MynahODDataset{},
			&model.MynahICDiagnosisReport{})

		if tableErr != nil {
			return nil, tableErr
		}

		syncErr := engine.Sync2(&model.MynahUser{},
			&model.MynahICProject{},
			&model.MynahFile{},
			&model.MynahICDataset{},
			&model.MynahODDataset{},
			&model.MynahICDiagnosisReport{})

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

// GetICProject get a project by id or return an error, second arg is requestor
func (d *localDB) GetICProject(uuid *string, requestor *model.MynahUser) (*model.MynahICProject, error) {
	project := model.MynahICProject{
		MynahProject: model.MynahProject{
			Uuid: *uuid,
		},
		Datasets: make([]string, 0),
		Reports:  make([]string, 0),
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

// GetODProject get a project by id or return an error, second arg is requestor
func (d *localDB) GetODProject(uuid *string, requestor *model.MynahUser) (*model.MynahODProject, error) {
	project := model.MynahODProject{
		MynahProject: model.MynahProject{
			Uuid: *uuid,
		},
		Datasets: make([]string, 0),
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
func (d *localDB) GetFiles(uuids []string, requestor *model.MynahUser) (map[string]*model.MynahFile, error) {
	var files []*model.MynahFile

	res := make(map[string]*model.MynahFile)

	//request a set of uuids within the org
	if err := d.engine.Where("org_id = ?", requestor.OrgId).In("uuid", uuids).Find(&files); err != nil {
		return nil, err
	}

	for _, f := range files {
		//check that the user has permission
		if commonErr := commonGetFile(f, requestor); commonErr == nil {
			//add to the filtered map
			res[f.Uuid] = f
		} else {
			log.Warnf("user %s failed to view file %s", requestor.Uuid, f.Uuid)
		}
	}

	return res, nil
}

// GetICDataset get a dataset from the database
func (d *localDB) GetICDataset(uuid *string, requestor *model.MynahUser) (*model.MynahICDataset, error) {
	dataset := model.MynahICDataset{
		MynahDataset: model.MynahDataset{
			Uuid: *uuid,
		},
		Files: make(map[string]*model.MynahICDatasetFile),
	}

	found, err := d.engine.Where("org_id = ?", requestor.OrgId).Get(&dataset)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("icdataset %s not found", *uuid)
	}

	//check that the user has permission
	if commonErr := commonGetDataset(&dataset, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &dataset, nil
}

// GetODDataset get a dataset from the database
func (d *localDB) GetODDataset(uuid *string, requestor *model.MynahUser) (*model.MynahODDataset, error) {
	dataset := model.MynahODDataset{
		MynahDataset: model.MynahDataset{
			Uuid: *uuid,
		},
		Entities:     make(map[string]*model.MynahODDatasetEntity),
		Files:        make(map[string]*model.MynahODDatasetFile),
		FileEntities: make(map[string][]string),
	}

	found, err := d.engine.Where("org_id = ?", requestor.OrgId).Get(&dataset)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("oddataset %s not found", *uuid)
	}

	//check that the user has permission
	if commonErr := commonGetDataset(&dataset, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &dataset, nil
}

// GetICDatasets get multiple ic datasets from the database
func (d *localDB) GetICDatasets(uuids []string, requestor *model.MynahUser) (map[string]*model.MynahICDataset, error) {
	var datasets []*model.MynahICDataset

	res := make(map[string]*model.MynahICDataset)

	//request a set of uuids within the org
	if err := d.engine.Where("org_id = ?", requestor.OrgId).In("uuid", uuids).Find(&datasets); err != nil {
		return nil, err
	}

	for _, d := range datasets {
		//check that the user has permission
		if commonErr := commonGetDataset(d, requestor); commonErr == nil {
			//add to the filtered map
			res[d.Uuid] = d
		} else {
			log.Warnf("user %s failed to view ic dataset %s", requestor.Uuid, d.Uuid)
		}
	}

	return res, nil
}

// GetODDatasets get multiple oc datasets from the database
func (d *localDB) GetODDatasets(uuids []string, requestor *model.MynahUser) (map[string]*model.MynahODDataset, error) {
	var datasets []*model.MynahODDataset

	res := make(map[string]*model.MynahODDataset)

	//request a set of uuids within the org
	if err := d.engine.Where("org_id = ?", requestor.OrgId).In("uuid", uuids).Find(&datasets); err != nil {
		return nil, err
	}

	for _, d := range datasets {
		//check that the user has permission
		if commonErr := commonGetDataset(d, requestor); commonErr == nil {
			//add to the filtered map
			res[d.Uuid] = d
		} else {
			log.Warnf("user %s failed to view oc dataset %s", requestor.Uuid, d.Uuid)
		}
	}

	return res, nil
}

// GetICDiagnosisReport get a diagnosis report
func (d *localDB) GetICDiagnosisReport(uuid *string, requestor *model.MynahUser) (*model.MynahICDiagnosisReport, error) {
	report := model.MynahICDiagnosisReport{
		MynahReport: model.MynahReport{
			Uuid: *uuid,
		},
	}

	found, err := d.engine.Where("org_id = ?", requestor.OrgId).Get(&report)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("report %s not found", *uuid)
	}

	//check that the user has permission
	if commonErr := commonGetReport(&report, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &report, nil
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

// ListICProjects list all projects, arg is requestor
func (d *localDB) ListICProjects(requestor *model.MynahUser) (projects []*model.MynahICProject, err error) {
	//list projects
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&projects)
	//filter for the projects that this user can view
	return commonListICProjects(projects, requestor), err
}

// ListODProjects list all projects, arg is requestor
func (d *localDB) ListODProjects(requestor *model.MynahUser) (projects []*model.MynahODProject, err error) {
	//list projects
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&projects)
	//filter for the projects that this user can view
	return commonListODProjects(projects, requestor), err
}

// ListFiles list all files, arg is requestor
func (d *localDB) ListFiles(requestor *model.MynahUser) (files []*model.MynahFile, err error) {
	//list files
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&files)
	//filter for the files that this user can view
	return commonListFiles(files, requestor), err
}

// ListICDatasets list all datasets, arg is requestor
func (d *localDB) ListICDatasets(requestor *model.MynahUser) (datasets []*model.MynahICDataset, err error) {
	//list datasets
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&datasets)
	//filter for the datasets that this user can view
	return commonListICDatasets(datasets, requestor), err
}

// ListODDatasets list all datasets, arg is requestor
func (d *localDB) ListODDatasets(requestor *model.MynahUser) (datasets []*model.MynahODDataset, err error) {
	//list datasets
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&datasets)
	//filter for the datasets that this user can view
	return commonListODDatasets(datasets, requestor), err
}

// CreateUser create a new user
func (d *localDB) CreateUser(creator *model.MynahUser, precommit func(*model.MynahUser) error) (*model.MynahUser, error) {
	user, err := commonCreateUser(creator)

	if err != nil {
		return nil, err
	}

	//call handler to make changes before commit
	if err := precommit(user); err != nil {
		return nil, err
	}

	affected, err := d.engine.Insert(user)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("user %s not created (no records affected)", user.Uuid)
	}
	return user, nil
}

// CreateICProject create a new project, second arg is creator
func (d *localDB) CreateICProject(creator *model.MynahUser, precommit func(*model.MynahICProject) error) (*model.MynahICProject, error) {
	project := commonCreateICProject(creator)

	if err := precommit(project); err != nil {
		return nil, err
	}

	affected, err := d.engine.Insert(project)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("icproject %s not created (no records affected)", project.Uuid)
	}
	return project, nil
}

// CreateODProject create a new project, arg is creator
func (d *localDB) CreateODProject(creator *model.MynahUser, precommit func(*model.MynahODProject) error) (*model.MynahODProject, error) {
	project := commonCreateODProject(creator)

	if err := precommit(project); err != nil {
		return nil, err
	}

	affected, err := d.engine.Insert(project)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("odproject %s not created (no records affected)", project.Uuid)
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

// CreateICDataset create a new dataset
func (d *localDB) CreateICDataset(creator *model.MynahUser, precommit func(*model.MynahICDataset) error) (*model.MynahICDataset, error) {
	dataset := commonCreateICDataset(creator)

	if err := precommit(dataset); err != nil {
		return nil, err
	}

	affected, err := d.engine.Insert(dataset)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("icdataset %s not created (no records affected)", dataset.Uuid)
	}
	return dataset, nil
}

// CreateODDataset create a new dataset
func (d *localDB) CreateODDataset(creator *model.MynahUser, precommit func(*model.MynahODDataset) error) (*model.MynahODDataset, error) {
	dataset := commonCreateODDataset(creator)

	if err := precommit(dataset); err != nil {
		return nil, err
	}

	affected, err := d.engine.Insert(dataset)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("oddataset %s not created (no records affected)", dataset.Uuid)
	}
	return dataset, nil
}

// CreateICDiagnosisReport creates a new ic diagnosis report in the database
func (d *localDB) CreateICDiagnosisReport(creator *model.MynahUser, precommit func(*model.MynahICDiagnosisReport) error) (*model.MynahICDiagnosisReport, error) {
	report := commonCreateICDiagnosisReport(creator)

	if err := precommit(report); err != nil {
		return nil, err
	}

	affected, err := d.engine.Insert(report)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("report %s not created (no records affected)", report.Uuid)
	}
	return report, nil
}

// UpdateUser update a user in the database
func (d *localDB) UpdateUser(user *model.MynahUser, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateUser(user, requestor, &keys); commonErr != nil {
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

// UpdateICProject update a project in the database
func (d *localDB) UpdateICProject(project *model.MynahICProject, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateProject(project, requestor, &keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(project)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("icproject %s not updated (no records affected)", project.Uuid)
	}
	return nil
}

// UpdateODProject update a project in the database. First arg is uuid of project to update, second is requestor, remaining
//are keys to update
func (d *localDB) UpdateODProject(project *model.MynahODProject, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateProject(project, requestor, &keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(project)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("odproject %s not updated (no records affected)", project.Uuid)
	}
	return nil
}

// UpdateICDataset update a dataset
func (d *localDB) UpdateICDataset(dataset *model.MynahICDataset, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateDataset(dataset, requestor, &keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("icdataset %s not updated (no records affected)", dataset.Uuid)
	}
	return nil
}

// UpdateODDataset update a dataset
func (d *localDB) UpdateODDataset(dataset *model.MynahODDataset, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateDataset(dataset, requestor, &keys); commonErr != nil {
		return commonErr
	}
	affected, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("oddataset %s not updated (no records affected)", dataset.Uuid)
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
		return fmt.Errorf("icproject %s not deleted (no records affected)", *uuid)
	}
	return nil
}

// DeleteODProject delete a project in the database, second arg is requestor
func (d *localDB) DeleteODProject(uuid *string, requestor *model.MynahUser) error {
	project, getErr := d.GetODProject(uuid, requestor)
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
		return fmt.Errorf("odproject %s not deleted (no records affected)", *uuid)
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
		return fmt.Errorf("icdataset %s not deleted (no records affected)", *uuid)
	}
	return nil
}

// DeleteODDataset delete a dataset
func (d *localDB) DeleteODDataset(uuid *string, requestor *model.MynahUser) error {
	dataset, getErr := d.GetODDataset(uuid, requestor)
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
		return fmt.Errorf("oddataset %s not deleted (no records affected)", *uuid)
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
