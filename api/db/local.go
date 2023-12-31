// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"errors"
	"fmt"
	_ "github.com/mattn/go-sqlite3"
	"os"
	"path/filepath"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db/engine"
	"reiform.com/mynah/db/migrations"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"xorm.io/xorm"
)

//local database client implements DBProvider
type localDB struct {
	engine.Engine
}

// create a string slice from a col name slice
func colsToStrings(cols []model.MynahColName) []string {
	strs := make([]string, len(cols))
	for i := 0; i < len(cols); i++ {
		strs[i] = string(cols[i])
	}
	return strs
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
		OrgId:   model.NewMynahUuid(),
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
	xormEngine, engineErr := xorm.NewEngine("sqlite3", mynahSettings.DBSettings.LocalPath)
	if engineErr != nil {
		return nil, engineErr
	}

	// set the log options
	xormEngine.ShowSQL(mynahSettings.DBSettings.LogSQL)

	db := localDB{
		engine.NewEngine(xormEngine),
	}

	if !exists {
		//create tables and initial orgs
		tableErr := xormEngine.CreateTables(&model.MynahUser{},
			&model.MynahFile{},
			&model.MynahICDataset{},
			&model.MynahODDataset{},
			&model.MynahICDatasetReport{},
			&model.MynahBinObject{})

		if tableErr != nil {
			return nil, tableErr
		}

		syncErr := xormEngine.Sync2(&model.MynahUser{},
			&model.MynahFile{},
			&model.MynahICDataset{},
			&model.MynahODDataset{},
			&model.MynahICDatasetReport{},
			&model.MynahBinObject{})

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

	// apply migrations
	return &db, migrations.Migrate(xormEngine)
}

// Transaction creates a new transaction limited to the execution of the handler
func (d *localDB) Transaction(handler func(DBProvider) error) error {
	return d.NewTransaction(func(e engine.Engine) error {
		// create a new provider using this transaction
		return handler(&localDB{
			e,
		})
	})
}

// GetUserForAuth Get a user by uuid or return an error
func (d *localDB) GetUserForAuth(uuid model.MynahUuid) (*model.MynahUser, error) {
	user := model.MynahUser{
		Uuid: uuid,
	}

	found, err := d.GetEngine().Get(&user)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("user %s not found", uuid)
	}
	return &user, nil
}

// GetUser Get a user other than self (must be admin)
func (d *localDB) GetUser(uuid model.MynahUuid, requestor *model.MynahUser) (*model.MynahUser, error) {
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

// GetFile get a file from the database
func (d *localDB) GetFile(uuid model.MynahUuid, requestor *model.MynahUser) (*model.MynahFile, error) {
	var file model.MynahFile

	found, err := d.GetEngine().Where("org_id = ?", requestor.OrgId).And("uuid = ?", uuid).Get(&file)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("file %s not found", uuid)
	}

	//check that the user has permission
	if commonErr := commonGetFile(&file, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &file, nil
}

// GetFiles get multiple files by id
func (d *localDB) GetFiles(uuids []model.MynahUuid, requestor *model.MynahUser) (model.MynahFileSet, error) {
	var files []*model.MynahFile

	res := make(model.MynahFileSet)

	//request a set of uuids within the org
	if err := d.GetEngine().Where("org_id = ?", requestor.OrgId).In("uuid", uuids).Find(&files); err != nil {
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
func (d *localDB) GetICDataset(uuid model.MynahUuid, requestor *model.MynahUser, omitCols ...model.MynahColName) (*model.MynahICDataset, error) {
	var dataset model.MynahICDataset

	found, err := d.GetEngine().Where("org_id = ?", requestor.OrgId).And("uuid = ?", uuid).Omit(colsToStrings(omitCols)...).Get(&dataset)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("icdataset %s not found", uuid)
	}

	//check that the user has permission
	if commonErr := commonGetDataset(&dataset, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &dataset, nil
}

// GetODDataset get a dataset from the database
func (d *localDB) GetODDataset(uuid model.MynahUuid, requestor *model.MynahUser) (*model.MynahODDataset, error) {
	var dataset model.MynahODDataset

	found, err := d.GetEngine().Where("org_id = ?", requestor.OrgId).And("uuid = ?", uuid).Get(&dataset)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("oddataset %s not found", uuid)
	}

	//check that the user has permission
	if commonErr := commonGetDataset(&dataset, requestor); commonErr != nil {
		return nil, commonErr
	}

	return &dataset, nil
}

// GetBinObject gets a cached object by uuid
func (d *localDB) GetBinObject(uuid model.MynahUuid, requestor *model.MynahUser) (*model.MynahBinObject, error) {
	var obj model.MynahBinObject

	found, err := d.GetEngine().Where("org_id = ?", requestor.OrgId).And("uuid = ?", uuid).Get(&obj)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("bin object %s not found", uuid)
	}

	return &obj, nil
}

// GetICDatasets get multiple ic datasets from the database
func (d *localDB) GetICDatasets(uuids []model.MynahUuid, requestor *model.MynahUser) (map[model.MynahUuid]*model.MynahICDataset, error) {
	var datasets []*model.MynahICDataset

	res := make(map[model.MynahUuid]*model.MynahICDataset)

	//request a set of uuids within the org
	if err := d.GetEngine().Where("org_id = ?", requestor.OrgId).In("uuid", uuids).Find(&datasets); err != nil {
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
func (d *localDB) GetODDatasets(uuids []model.MynahUuid, requestor *model.MynahUser) (map[model.MynahUuid]*model.MynahODDataset, error) {
	var datasets []*model.MynahODDataset

	res := make(map[model.MynahUuid]*model.MynahODDataset)

	//request a set of uuids within the org
	if err := d.GetEngine().Where("org_id = ?", requestor.OrgId).In("uuid", uuids).Find(&datasets); err != nil {
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

// ListUsers list all users in an org
func (d *localDB) ListUsers(requestor *model.MynahUser) (users []*model.MynahUser, err error) {
	if commonErr := commonListUsers(requestor); commonErr != nil {
		return users, commonErr
	}

	//list users
	err = d.GetEngine().Where("org_id = ?", requestor.OrgId).Find(&users)
	return users, err
}

// ListFiles list all files, arg is requestor
func (d *localDB) ListFiles(requestor *model.MynahUser) (files []*model.MynahFile, err error) {
	//list files
	err = d.GetEngine().Where("org_id = ?", requestor.OrgId).Find(&files)
	//filter for the files that this user can view
	return commonListFiles(files, requestor), err
}

// ListICDatasets list all datasets, arg is requestor
func (d *localDB) ListICDatasets(requestor *model.MynahUser) (datasets []*model.MynahICDataset, err error) {
	//list datasets
	err = d.GetEngine().Where("org_id = ?", requestor.OrgId).Find(&datasets)
	//filter for the datasets that this user can view
	return commonListICDatasets(datasets, requestor), err
}

// ListODDatasets list all datasets, arg is requestor
func (d *localDB) ListODDatasets(requestor *model.MynahUser) (datasets []*model.MynahODDataset, err error) {
	//list datasets
	err = d.GetEngine().Where("org_id = ?", requestor.OrgId).Find(&datasets)
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

	affected, err := d.GetEngine().Insert(user)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("user %s not created (no records affected)", user.Uuid)
	}
	return user, nil
}

// CreateFile create a new file, second arg is creator
func (d *localDB) CreateFile(creator *model.MynahUser, precommit func(*model.MynahFile) error) (*model.MynahFile, error) {
	file := model.NewFile(creator)

	//since we can't update files once created, we may need to fail during creation when writing to local storage
	if err := precommit(file); err != nil {
		return nil, err
	}

	affected, err := d.GetEngine().Insert(file)
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
	dataset := model.NewICDataset(creator)

	if err := precommit(dataset); err != nil {
		return nil, err
	}

	affected, err := d.GetEngine().Insert(dataset)
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
	dataset := model.NewODDataset(creator)

	if err := precommit(dataset); err != nil {
		return nil, err
	}

	affected, err := d.GetEngine().Insert(dataset)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("oddataset %s not created (no records affected)", dataset.Uuid)
	}
	return dataset, nil
}

// CreateBinObject stores some binary data in the database
func (d *localDB) CreateBinObject(creator *model.MynahUser, precommit func(*model.MynahBinObject) error) (*model.MynahBinObject, error) {
	obj := model.NewBinObject(creator)

	if err := precommit(obj); err != nil {
		return nil, err
	}

	affected, err := d.GetEngine().Insert(obj)
	if err != nil {
		return nil, err
	}
	if affected == 0 {
		return nil, fmt.Errorf("bin data %s not created (no records affected)", obj.Uuid)
	}
	return obj, nil
}

// UpdateUser update a user in the database
func (d *localDB) UpdateUser(user *model.MynahUser, requestor *model.MynahUser, cols ...model.MynahColName) error {
	if commonErr := commonUpdateUser(user, requestor, cols); commonErr != nil {
		return commonErr
	}

	affected, err := d.GetEngine().Where("org_id = ?", requestor.OrgId).And("uuid = ?", user.Uuid).Cols(colsToStrings(cols)...).Update(user)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("user %s not updated (no records affected)", user.Uuid)
	}
	return nil
}

// UpdateICDataset update a dataset
func (d *localDB) UpdateICDataset(dataset *model.MynahICDataset, requestor *model.MynahUser, cols ...model.MynahColName) error {
	cols = append(cols, model.DateModifiedCol)
	if commonErr := commonUpdateDataset(dataset, requestor, cols); commonErr != nil {
		return commonErr
	}

	affected, err := d.GetEngine().Where("org_id = ?", requestor.OrgId).And("uuid = ?", dataset.Uuid).Cols(colsToStrings(cols)...).Update(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("icdataset %s not updated (no records affected)", dataset.Uuid)
	}
	return nil
}

// UpdateODDataset update a dataset
func (d *localDB) UpdateODDataset(dataset *model.MynahODDataset, requestor *model.MynahUser, cols ...model.MynahColName) error {
	cols = append(cols, model.DateModifiedCol)
	if commonErr := commonUpdateDataset(dataset, requestor, cols); commonErr != nil {
		return commonErr
	}

	affected, err := d.GetEngine().Where("org_id = ?", requestor.OrgId).And("uuid = ?", dataset.Uuid).Cols(colsToStrings(cols)...).Update(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("oddataset %s not updated (no records affected)", dataset.Uuid)
	}
	return nil
}

// UpdateFile updates a file
func (d *localDB) UpdateFile(file *model.MynahFile, requestor *model.MynahUser, cols ...model.MynahColName) error {
	if commonErr := commonUpdateFile(file, requestor, cols); commonErr != nil {
		return commonErr
	}

	affected, err := d.GetEngine().Where("org_id = ?", requestor.OrgId).And("uuid = ?", file.Uuid).Cols(colsToStrings(cols)...).Update(file)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("file %s not updated (no records affected)", file.Uuid)
	}
	return nil
}

// UpdateFiles updates a set of files.
func (d *localDB) UpdateFiles(files model.MynahFileSet, requestor *model.MynahUser, cols ...model.MynahColName) error {
	return d.Transaction(func(db DBProvider) error {
		for _, file := range files {
			err := db.UpdateFile(file, requestor, cols...)
			if err != nil {
				return fmt.Errorf("error updating file %s: %s", file.Uuid, err)
			}
		}

		return nil
	})
}

// DeleteUser delete a user in the database
func (d *localDB) DeleteUser(uuid model.MynahUuid, requestor *model.MynahUser) error {
	if commonErr := commonDeleteUser(uuid, requestor); commonErr != nil {
		return commonErr
	}
	affected, err := d.GetEngine().Delete(&model.MynahUser{Uuid: uuid})
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("user %s not deleted (no records affected)", uuid)
	}
	return nil
}

// DeleteFile delete a file in the database, second arg is requestor
func (d *localDB) DeleteFile(uuid model.MynahUuid, requestor *model.MynahUser) error {
	file, getErr := d.GetFile(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the file to check permissions
	if commonErr := commonDeleteFile(file, requestor); commonErr != nil {
		return commonErr
	}

	affected, err := d.GetEngine().Delete(file)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("file %s not deleted (no records affected)", uuid)
	}
	return nil
}

// DeleteICDataset delete a dataset
func (d *localDB) DeleteICDataset(uuid model.MynahUuid, requestor *model.MynahUser) error {
	dataset, getErr := d.GetICDataset(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the dataset to check permissions
	if commonErr := commonDeleteDataset(dataset, requestor); commonErr != nil {
		return commonErr
	}

	//delete referenced reports
	for _, reportMetadata := range dataset.Reports {
		affected, err := d.GetEngine().Delete(&model.MynahBinObject{
			Uuid: reportMetadata.DataId,
		})
		if err != nil {
			log.Warnf("when deleting ic dataset %s, report %s was not deleted: %s", uuid, reportMetadata.DataId, err)
		}
		if affected == 0 {
			log.Warnf("when deleting ic dataset %s, report %s was not deleted, no records affected", uuid, reportMetadata.DataId)
		}
	}

	affected, err := d.GetEngine().Delete(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("icdataset %s not deleted (no records affected)", uuid)
	}
	return nil
}

// DeleteODDataset delete a dataset
func (d *localDB) DeleteODDataset(uuid model.MynahUuid, requestor *model.MynahUser) error {
	dataset, getErr := d.GetODDataset(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the dataset to check permissions
	if commonErr := commonDeleteDataset(dataset, requestor); commonErr != nil {
		return commonErr
	}

	affected, err := d.GetEngine().Delete(dataset)
	if err != nil {
		return err
	}
	if affected == 0 {
		return fmt.Errorf("oddataset %s not deleted (no records affected)", uuid)
	}
	return nil
}

// Close the client connection on shutdown
func (d *localDB) Close() {
	log.Infof("local database engine shutdown")
	if err := d.CloseEngine(); err != nil {
		log.Warnf("error closing database: %s", err)
	}
}
