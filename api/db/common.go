package db

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"reiform.com/mynah/model"
	"strconv"
)

//create a sql statement and convert values to sql types
func commonCreateSQLUpdateStmt(s model.Identity, tableName string, keys []string) (stmt *string, vals []interface{}, e error) {
	//create the update statement
	updateStmt := fmt.Sprintf("UPDATE %s SET ", tableName)
	for i, k := range keys {
		//perform orm mapping
		if sqlVal, ormErr := sqlORM(s, k); ormErr == nil {
			//add to update statement
			if i < (len(keys) - 1) {
				updateStmt += fmt.Sprintf("%s = ?, ", k)
			} else {
				updateStmt += fmt.Sprintf("%s = ? ", k)
			}

			vals = append(vals, sqlVal)
		} else {
			return nil, nil, ormErr
		}
	}

	updateStmt += " WHERE Uuid = ? AND OrgId = ?"
	vals = append(vals, s.GetUuid(), s.GetOrgId())
	return &updateStmt, vals, nil
}

//check if any of the update keys are restricted
func restrictedKeys(keys []string) bool {
	for _, s := range keys {
		if (s == "org_id") || (s == "uuid") {
			return true
		}
	}
	return false
}

//scan a SQL row into a user
func scanRowUser(rows *sql.Rows) (*model.MynahUser, error) {
	var u model.MynahUser
	//need to parse string back to bool
	var isAdmin string

	//load the values from the row into the new user struct
	if scanErr := rows.Scan(&u.Uuid, &u.OrgId, &u.NameFirst, &u.NameLast, &isAdmin, &u.CreatedBy); scanErr != nil {
		return nil, scanErr
	}

	//parse the admin flag
	if b, bErr := strconv.ParseBool(isAdmin); bErr == nil {
		u.IsAdmin = b
	} else {
		log.Printf("incorrectly parsed admin flag (%s) for user %s", isAdmin, u.Uuid)
		u.IsAdmin = false
	}

	return &u, nil
}

//scan a SQL result row into a Mynah project struct
func scanRowProject(rows *sql.Rows) (*model.MynahProject, error) {
	var p model.MynahProject
	p.UserPermissions = make(map[string]model.ProjectPermissions)

	//need to parse string back to map
	var projectPermissions string

	//load the values from the row into the new project struct
	if scanErr := rows.Scan(&p.Uuid, &p.OrgId, &projectPermissions, &p.ProjectName); scanErr != nil {
		return nil, scanErr
	}

	//parse the permissions
	if err := deserializeJson(&projectPermissions, &p.UserPermissions); err != nil {
		return nil, err
	}

	return &p, nil
}

//scan a SQL result row into a Mynah file struct
func scanRowFile(rows *sql.Rows) (*model.MynahFile, error) {
	var f model.MynahFile

	//load the values from the row into the new file struct
	if scanErr := rows.Scan(&f.Uuid, &f.OrgId, &f.OwnerUuid, &f.Name, &f.Location, &f.Path); scanErr != nil {
		return nil, scanErr
	}
	return &f, nil
}

//serialize some structure into a string
func serializeJson(jsonStruct interface{}) (*string, error) {
	if b, err := json.Marshal(jsonStruct); err == nil {
		s := string(b)
		return &s, nil
	} else {
		return nil, err
	}
}

//deserialize a string back into some json structure
func deserializeJson(jsonRep *string, target interface{}) error {
	return json.Unmarshal([]byte(*jsonRep), target)
}

//verify that the requestor is an admin
func commonGetUser(user *model.MynahUser, requestor *model.MynahUser) error {
	//Note: this is the only location where we need to compare org id (it isn't filtered in the query
	//so that auth works with no knowledge of org)
	if (user.OrgId == requestor.OrgId) && requestor.IsAdmin {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request user %s", requestor.Uuid, user.Uuid)
}

//get a project by id or return an error
func commonGetProject(project *model.MynahProject, requestor *model.MynahUser) error {
	//check that the user has permission to at least view this project
	if requestor.IsAdmin || project.GetPermissions(requestor) >= model.Read {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request project %s", requestor.Uuid, project.Uuid)
}

//get a project by id or return an error
func commonGetFile(file *model.MynahFile, requestor *model.MynahUser) error {
	//check that the user is the file owner (or admin)
	if requestor.IsAdmin || requestor.Uuid == file.OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request file %s", requestor.Uuid, file.Uuid)
}

//check that the user is an admin (org checked in query)
func commonListUsers(requestor *model.MynahUser) error {
	if requestor.IsAdmin {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to list users", requestor.Uuid)
}

//get the projects that the user can view
func commonListProjects(projects []*model.MynahProject, requestor *model.MynahUser) (filtered []*model.MynahProject) {
	//filter for projects that this user has permission to view
	for _, p := range projects {
		if e := commonGetProject(p, requestor); e == nil {
			filtered = append(filtered, p)
		}
	}
	return filtered
}

//get the files that the user can view
func commonListFiles(files []*model.MynahFile, requestor *model.MynahUser) (filtered []*model.MynahFile) {
	//filter for files that this user has permission to view
	for _, f := range files {
		if e := commonGetFile(f, requestor); e == nil {
			filtered = append(filtered, f)
		}
	}
	return filtered
}

//check that the creator is an admin
func commonCreateUser(user *model.MynahUser, creator *model.MynahUser) error {
	if !creator.IsAdmin {
		return fmt.Errorf("unable to create new user, user %s is not an admin", creator.Uuid)
	}
	if user.Uuid == creator.Uuid {
		return errors.New("user must have a distinct creator")
	}

	//set the creator uuid
	user.CreatedBy = creator.Uuid
	//inherit the org id
	user.OrgId = creator.OrgId
	return nil
}

//create a new project
func commonCreateProject(project *model.MynahProject, creator *model.MynahUser) error {
	//give ownership permissions to the user
	project.UserPermissions[creator.Uuid] = model.Owner
	//inherit the org id
	project.OrgId = creator.OrgId
	return nil
}

//create a new file
func commonCreateFile(file *model.MynahFile, creator *model.MynahUser) error {
	//give ownership to the user
	file.OwnerUuid = creator.Uuid
	//inherit the org id
	file.OrgId = creator.OrgId
	return nil
}

//update a user in the database
func commonUpdateUser(user *model.MynahUser, requestor *model.MynahUser, keys []string) error {
	//check that keys are not restricted
	if restrictedKeys(keys) {
		return errors.New("user update contained restricted keys")
	}

	if requestor.IsAdmin || requestor.Uuid == user.Uuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update user %s", requestor.Uuid, user.Uuid)
}

//update a project in the database
func commonUpdateProject(project *model.MynahProject, requestor *model.MynahUser, keys []string) error {
	//check that keys are not restricted
	if restrictedKeys(keys) {
		return errors.New("project update contained restricted keys")
	}

	if requestor.IsAdmin || project.GetPermissions(requestor) >= model.Read {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update project %s", requestor.Uuid, project.Uuid)
}

//update a file in the database
func commonUpdateFile(file *model.MynahFile, requestor *model.MynahUser, keys []string) error {
	//check that keys are not restricted
	if restrictedKeys(keys) {
		return errors.New("file update contained restricted keys")
	}

	if requestor.IsAdmin || requestor.Uuid == file.OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update file %s", requestor.Uuid, file.Uuid)
}

//check that the requestor has permission
func commonDeleteUser(uuid *string, requestor *model.MynahUser) error {
	if requestor.IsAdmin {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update user %s", requestor.Uuid, *uuid)
}

//check that the requestor has permission to delete the project
func commonDeleteProject(project *model.MynahProject, requestor *model.MynahUser) error {
	if requestor.IsAdmin || project.GetPermissions(requestor) == model.Owner {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to delete project %s", requestor.Uuid, project.Uuid)
}

//check that the requestor has permission to delete the file
func commonDeleteFile(file *model.MynahFile, requestor *model.MynahUser) error {
	if requestor.IsAdmin || requestor.Uuid == file.OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to delete file %s", requestor.Uuid, file.Uuid)
}
