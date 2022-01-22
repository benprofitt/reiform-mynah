package db

import (
	"errors"
	"fmt"
	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"
	"log"
	"os"
	"reiform.com/mynah/auth"
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
		if file, err := os.Create(path); err == nil {
			file.Close()
			return false, nil
		} else {
			return false, err
		}
	} else {
		return false, fmt.Errorf("failed to identify whether database already exists: %s", err)
	}
}

//create a new organization in the database and a starting admin user
func (d *localDB) createLocalOrg(authProvider auth.AuthProvider) error {
	//create the initial admin user, organization id
	admin, jwt, adminErr := authProvider.CreateUser()
	if adminErr != nil {
		return adminErr
	}

	//set as admin and assign a new organization id
	admin.IsAdmin = true
	admin.OrgId = uuid.New().String()

	tempAdmin := model.MynahUser{}
	tempAdmin.OrgId = admin.OrgId
	tempAdmin.IsAdmin = true

	//log the initial information
	log.Printf("created organization %s", admin.OrgId)
	log.Printf("created initial admin JWT for org (%s): %s", admin.OrgId, jwt)

	//add the initial admin user into the database
	if createAdminErr := d.CreateUser(admin, &tempAdmin); createAdminErr != nil {
		return createAdminErr
	}
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
			&model.MynahProject{},
			&model.MynahFile{},
			&model.MynahDataset{})

		if tableErr != nil {
			return nil, tableErr
		}

		syncErr := engine.Sync2(&model.MynahUser{},
			&model.MynahProject{},
			&model.MynahFile{},
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

		log.Printf("created local database %s", mynahSettings.DBSettings.LocalPath)
	}

	return &db, nil
}

//Get a user by uuid or return an error
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

//Get a user other than self (must be admin)
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

//get a project by id or return an error
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

//get a file from the database
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

//get a dataset from the database
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

//list all users
func (d *localDB) ListUsers(requestor *model.MynahUser) (users []*model.MynahUser, err error) {
	if commonErr := commonListUsers(requestor); commonErr != nil {
		return users, commonErr
	}

	//list users
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&users)
	return users, err
}

//list all projects
func (d *localDB) ListProjects(requestor *model.MynahUser) (projects []*model.MynahProject, err error) {
	//list projects
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&projects)
	//filter for the projects that this user can view
	return commonListProjects(projects, requestor), err
}

//list all files, arg is requestor
func (d *localDB) ListFiles(requestor *model.MynahUser) (files []*model.MynahFile, err error) {
	//list files
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&files)
	//filter for the files that this user can view
	return commonListFiles(files, requestor), err
}

//list all datasets, arg is requestor
func (d *localDB) ListDatasets(requestor *model.MynahUser) (datasets []*model.MynahDataset, err error) {
	//list datasets
	err = d.engine.Where("org_id = ?", requestor.OrgId).Find(&datasets)
	//filter for the datasets that this user can view
	return commonListDatasets(datasets, requestor), err
}

//create a new user
func (d *localDB) CreateUser(user *model.MynahUser, creator *model.MynahUser) error {
	if commonErr := commonCreateUser(user, creator); commonErr != nil {
		return commonErr
	}

	_, err := d.engine.Insert(user)
	return err
}

//create a new project
func (d *localDB) CreateProject(project *model.MynahProject, creator *model.MynahUser) error {
	if commonErr := commonCreateProject(project, creator); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Insert(project)
	return err
}

//create a new file, second arg is creator
func (d *localDB) CreateFile(file *model.MynahFile, creator *model.MynahUser) error {
	if commonErr := commonCreateFile(file, creator); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Insert(file)
	return err
}

//create a new dataset
func (d *localDB) CreateDataset(dataset *model.MynahDataset, creator *model.MynahUser) error {
	if commonErr := commonCreateDataset(dataset, creator); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Insert(dataset)
	return err
}

//update a user in the database
func (d *localDB) UpdateUser(user *model.MynahUser, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateUser(user, requestor, keys); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(user)
	return err
}

//update a project in the database
func (d *localDB) UpdateProject(project *model.MynahProject, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateProject(project, requestor, keys); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(project)
	return err
}

//update a file in the database, second arg is requestor
func (d *localDB) UpdateFile(file *model.MynahFile, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateFile(file, requestor, keys); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(file)
	return err
}

//update a dataset
func (d *localDB) UpdateDataset(dataset *model.MynahDataset, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateDataset(dataset, requestor, keys); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Where("org_id = ?", requestor.OrgId).Cols(keys...).Update(dataset)
	return err
}

//delete a user in the database
func (d *localDB) DeleteUser(uuid *string, requestor *model.MynahUser) error {
	if commonErr := commonDeleteUser(uuid, requestor); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Delete(&model.MynahUser{Uuid: *uuid})
	return err
}

//delete a project in the database
func (d *localDB) DeleteProject(uuid *string, requestor *model.MynahUser) error {
	project, getErr := d.GetProject(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the project to check permissions
	if commonErr := commonDeleteProject(project, requestor); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Delete(project)
	return err
}

//delete a file in the database, second arg is requestor
func (d *localDB) DeleteFile(uuid *string, requestor *model.MynahUser) error {
	file, getErr := d.GetFile(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the project to check permissions
	if commonErr := commonDeleteFile(file, requestor); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Delete(file)
	return err
}

//delete a dataset
func (d *localDB) DeleteDataset(uuid *string, requestor *model.MynahUser) error {
	dataset, getErr := d.GetDataset(uuid, requestor)
	if getErr != nil {
		return getErr
	}
	//get the project to check permissions
	if commonErr := commonDeleteDataset(dataset, requestor); commonErr != nil {
		return commonErr
	}
	_, err := d.engine.Delete(dataset)
	return err
}

//close the client connection on shutdown
func (d *localDB) Close() {
	log.Printf("local database engine shutdown")
	d.engine.Close()
}
