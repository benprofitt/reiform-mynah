package db

import (
	"database/sql"
	"errors"
	"fmt"
	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"
	"log"
	"os"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
	"strconv"
)

//check if the local mynah database already exists
func localDBExists(path *string) bool {
	if _, err := os.Stat(*path); err == nil {
		return true
	} else if errors.Is(err, os.ErrNotExist) {
		return false
	} else {
		log.Fatalf("failed to identify whether database already exists: %s", err)
		return false
	}
}

//prepare and execute a sql statement
func (d *localDB) localPrepareExec(statement string, args ...interface{}) error {
	if statement, err := d.db.Prepare(statement); err == nil {
		if len(args) > 0 {
			if _, execErr := statement.Exec(args...); execErr != nil {
				return execErr
			}
		} else {
			if _, execErr := statement.Exec(); execErr != nil {
				return execErr
			}
		}
		return nil
	} else {
		return err
	}
}

//create the mynah local database
func createLocalDatabase(path *string) (*localDB, error) {
	//create the path
	if file, err := os.Create(*path); err == nil {
		file.Close()
	} else {
		return nil, err
	}

	//open the database
	sqlDB, err := sql.Open("sqlite3", *path)
	if err != nil {
		return nil, err
	}

	var ldb localDB
	ldb.db = sqlDB

	//create user table
	if userTableErr := ldb.localPrepareExec(createUserTableSQL); userTableErr != nil {
		return nil, userTableErr
	}

	//create the projects table
	if projectTableErr := ldb.localPrepareExec(createProjectTableSQL); projectTableErr != nil {
		return nil, projectTableErr
	}

	//create the files table
	if fileTableErr := ldb.localPrepareExec(createFileTableSQL); fileTableErr != nil {
		return nil, fileTableErr
	}

	return &ldb, nil
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

	//log the initial information
	log.Printf("created organization %s", admin.OrgId)
	log.Printf("created initial admin JWT for org (%s): %s", admin.OrgId, jwt)

	//add the initial admin user into the database
	if createAdminErr := d.CreateUser(admin, admin); createAdminErr != nil {
		return createAdminErr
	}
	return nil
}

//create a new local db instance
func newLocalDB(mynahSettings *settings.MynahSettings, authProvider auth.AuthProvider) (*localDB, error) {
	//only create database if it doesn't exist
	if !localDBExists(&mynahSettings.DBSettings.LocalPath) {
		db, err := createLocalDatabase(&mynahSettings.DBSettings.LocalPath)
		if err != nil {
			return nil, err
		}
		log.Printf("created local database %s", mynahSettings.DBSettings.LocalPath)

		//create initial organization structure
		for i := 0; i < mynahSettings.DBSettings.InitialOrgCount; i++ {
			db.createLocalOrg(authProvider)
		}

		return db, nil
	} else {
		//open the database
		sqlDB, err := sql.Open("sqlite3", mynahSettings.DBSettings.LocalPath)
		if err != nil {
			return nil, err
		}
		return &localDB{
			db: sqlDB,
		}, nil
	}
}

//Get a user by uuid or return an error
func (d *localDB) GetUserForAuth(uuid *string) (*model.MynahUser, error) {
	rows, err := d.db.Query(getUserSQL, *uuid)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		if user, userErr := scanRowUser(rows); userErr == nil {
			//there should only be one
			return user, nil

		} else {
			return nil, userErr
		}
	}

	return nil, errors.New(fmt.Sprintf("user %s not found", *uuid))
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
	rows, err := d.db.Query(getProjectSQL, *uuid)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		if project, projectErr := scanRowProject(rows); projectErr == nil {
			//verify that this user has permission
			if commonErr := commonGetProject(project, requestor); commonErr != nil {
				return nil, commonErr
			}
			//there should only be one
			return project, nil

		} else {
			return nil, projectErr
		}
	}

	return nil, errors.New(fmt.Sprintf("project %s not found", *uuid))
}

//get a file from the database
func (d *localDB) GetFile(uuid *string, requestor *model.MynahUser) (*model.MynahFile, error) {
	rows, err := d.db.Query(getFileSQL, *uuid)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		if file, fileErr := scanRowFile(rows); fileErr == nil {
			//verify that this user has permission
			if commonErr := commonGetFile(file, requestor); commonErr != nil {
				return nil, commonErr
			}
			//there should only be one
			return file, nil

		} else {
			return nil, fileErr
		}
	}

	return nil, errors.New(fmt.Sprintf("file %s not found", *uuid))
}

//list all users
func (d *localDB) ListUsers(requestor *model.MynahUser) (users []*model.MynahUser, err error) {
	if commonErr := commonListUsers(requestor); commonErr != nil {
		return users, commonErr
	}

	rows, queryErr := d.db.Query(listUsersSQL, requestor.OrgId)
	if queryErr != nil {
		return users, queryErr
	}
	defer rows.Close()

	for rows.Next() {
		if user, userErr := scanRowUser(rows); userErr == nil {
			users = append(users, user)
		} else {
			return users, userErr
		}
	}

	return users, err
}

//list all projects
func (d *localDB) ListProjects(requestor *model.MynahUser) (projects []*model.MynahProject, err error) {
	//request projects in org
	rows, queryErr := d.db.Query(listProjectsSQL, requestor.OrgId)
	if queryErr != nil {
		return projects, queryErr
	}
	defer rows.Close()

	//temporary list (not filtered by permissions the user has for each)
	tempProjects := make([]*model.MynahProject, 0)

	//scan projects into structs
	for rows.Next() {
		if project, projectErr := scanRowProject(rows); projectErr == nil {
			tempProjects = append(tempProjects, project)
		} else {
			return projects, projectErr
		}
	}

	//filter for the projects that this user can view
	return commonListProjects(tempProjects, requestor), nil
}

//list all files, arg is requestor
func (d *localDB) ListFiles(requestor *model.MynahUser) (files []*model.MynahFile, err error) {
	//request files in org
	rows, queryErr := d.db.Query(listFilesSQL, requestor.OrgId)
	if queryErr != nil {
		return files, queryErr
	}
	defer rows.Close()

	//temporary list (not filtered by permissions the user has for each)
	tempFiles := make([]*model.MynahFile, 0)

	//scan files into structs
	for rows.Next() {
		if file, fileErr := scanRowFile(rows); fileErr == nil {
			tempFiles = append(tempFiles, file)
		} else {
			//empty
			return files, fileErr
		}
	}

	//filter for the files that this user can view (must be admin or file owner)
	return commonListFiles(tempFiles, requestor), nil
}

//create a new user
func (d *localDB) CreateUser(user *model.MynahUser, creator *model.MynahUser) error {
	if commonErr := commonCreateUser(user, creator); commonErr != nil {
		return commonErr
	}
	//add a user to the table
	return d.localPrepareExec(createUserSQL,
		user.Uuid,
		user.OrgId,
		user.NameFirst,
		user.NameLast,
		strconv.FormatBool(user.IsAdmin),
		user.CreatedBy)
}

//create a new project
func (d *localDB) CreateProject(project *model.MynahProject, creator *model.MynahUser) error {
	if commonErr := commonCreateProject(project, creator); commonErr != nil {
		return commonErr
	}

	//serialize the project permissions
	stringProjectPerm, jsonErr := serializeJson(&project.UserPermissions)
	if jsonErr != nil {
		return jsonErr
	}

	//add a project to the table
	return d.localPrepareExec(createProjectSQL,
		project.Uuid,
		project.OrgId,
		stringProjectPerm,
		project.ProjectName)
}

//create a new file, second arg is creator
func (d *localDB) CreateFile(file *model.MynahFile, creator *model.MynahUser) error {
	if commonErr := commonCreateFile(file, creator); commonErr != nil {
		return commonErr
	}

	//add a project to the table
	return d.localPrepareExec(createFileSQL,
		file.Uuid,
		file.OrgId,
		file.OwnerUuid,
		file.Name,
		file.Location,
		file.Path)
}

//update a user in the database
func (d *localDB) UpdateUser(user *model.MynahUser, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateUser(user, requestor, keys); commonErr != nil {
		return commonErr
	}

	//create the update statement
	if stmt, vals, err := commonCreateSQLUpdateStmt(user, "user", keys); err == nil {
		return d.localPrepareExec(*stmt, vals...)
	} else {
		return err
	}
}

//update a project in the database
func (d *localDB) UpdateProject(project *model.MynahProject, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateProject(project, requestor, keys); commonErr != nil {
		return commonErr
	}

	//create the update statement
	if stmt, vals, err := commonCreateSQLUpdateStmt(project, "project", keys); err == nil {
		return d.localPrepareExec(*stmt, vals...)
	} else {
		return err
	}
}

//update a file in the database, second arg is requestor
func (d *localDB) UpdateFile(file *model.MynahFile, requestor *model.MynahUser, keys ...string) error {
	if commonErr := commonUpdateFile(file, requestor, keys); commonErr != nil {
		return commonErr
	}

	//create the update statement
	if stmt, vals, err := commonCreateSQLUpdateStmt(file, "file", keys); err == nil {
		return d.localPrepareExec(*stmt, vals...)
	} else {
		return err
	}
}

//delete a user in the database
func (d *localDB) DeleteUser(uuid *string, requestor *model.MynahUser) error {
	if commonErr := commonDeleteUser(uuid, requestor); commonErr != nil {
		return commonErr
	}

	//execute the request (uses requestor's org id to verify)
	return d.localPrepareExec(deleteUserSQL,
		*uuid,
		requestor.OrgId)
}

//delete a project in the database
func (d *localDB) DeleteProject(uuid *string, requestor *model.MynahUser) error {
	//need to first load the project to check if the user has permission
	if project, projectErr := d.GetProject(uuid, requestor); projectErr == nil {
		//validate that this user has permission to delete this project
		if commonErr := commonDeleteProject(project, requestor); commonErr != nil {
			return commonErr
		}
		//can delete
		return d.localPrepareExec(deleteProjectSQL,
			*uuid,
			project.OrgId)

	} else {
		return projectErr
	}
}

//delete a file in the database, second arg is requestor
func (d *localDB) DeleteFile(uuid *string, requestor *model.MynahUser) error {
	//need to first load the file to check if the user has permission
	if file, fileErr := d.GetFile(uuid, requestor); fileErr == nil {
		//validate that this user has permission to delete this file
		if commonErr := commonDeleteFile(file, requestor); commonErr != nil {
			return commonErr
		}
		//can delete
		return d.localPrepareExec(deleteFileSQL,
			*uuid,
			file.OrgId)

	} else {
		return fileErr
	}
}

//close the client connection on shutdown
func (d *localDB) Close() {
	log.Printf("shutdown")
	d.db.Close()
}
