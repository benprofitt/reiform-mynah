package db

import (
	"encoding/json"
	"errors"
	"fmt"
	"reiform.com/mynah/model"
	"github.com/google/uuid"
)

//check if any of the update keys are restricted
func restrictedKeys(keys []string) bool {
	for _, s := range keys {
		if (s == "org_id") || (s == "uuid") {
			return true
		}
	}
	return false
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

//get a file by id or return an error
func commonGetFile(file *model.MynahFile, requestor *model.MynahUser) error {
	//check that the user is the file owner (or admin)
	if requestor.IsAdmin || requestor.Uuid == file.OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request file %s", requestor.Uuid, file.Uuid)
}

//get a dataset by id or return an error
func commonGetDataset(dataset *model.MynahDataset, requestor *model.MynahUser) error {
	//check that the user is the dataset owner (or admin)
	if requestor.IsAdmin || requestor.Uuid == dataset.OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request dataset %s", requestor.Uuid, dataset.Uuid)
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

//get the datasets that the user can view
func commonListDatasets(datasets []*model.MynahDataset, requestor *model.MynahUser) (filtered []*model.MynahDataset) {
	//filter for files that this user has permission to view
	for _, d := range datasets {
		if e := commonGetDataset(d, requestor); e == nil {
			filtered = append(filtered, d)
		}
	}
	return filtered
}

//check that the creator is an admin
func commonCreateUser(user *model.MynahUser, creator *model.MynahUser) error {
	//Note: uuid created by auth provider
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
	project.Uuid = uuid.New().String()
	return nil
}

//create a new file
func commonCreateFile(file *model.MynahFile, creator *model.MynahUser) error {
	//give ownership to the user
	file.OwnerUuid = creator.Uuid
	//inherit the org id
	file.OrgId = creator.OrgId
	file.Uuid = uuid.New().String()
	return nil
}

//create a new dataset
func commonCreateDataset(dataset *model.MynahDataset, creator *model.MynahUser) error {
	//give ownership to the user
	dataset.OwnerUuid = creator.Uuid
	//inherit the org id
	dataset.OrgId = creator.OrgId
	dataset.Uuid = uuid.New().String()
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

//update a dataset in the database
func commonUpdateDataset(dataset *model.MynahDataset, requestor *model.MynahUser, keys []string) error {
	//check that keys are not restricted
	if restrictedKeys(keys) {
		return errors.New("dataset update contained restricted keys")
	}

	if requestor.IsAdmin || requestor.Uuid == dataset.OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update dataset %s", requestor.Uuid, dataset.Uuid)
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
	//TODO check if file is part of dataset
	if requestor.IsAdmin || requestor.Uuid == file.OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to delete file %s", requestor.Uuid, file.Uuid)
}

//check that the requestor has permission to delete the dataset
func commonDeleteDataset(dataset *model.MynahDataset, requestor *model.MynahUser) error {
	if requestor.IsAdmin || requestor.Uuid == dataset.OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to delete dataset %s", requestor.Uuid, dataset.Uuid)
}
