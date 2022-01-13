package db

import (
	"encoding/json"
	"errors"
	"fmt"
	"reiform.com/mynah/model"
)

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
	return errors.New(fmt.Sprintf("user %s does not have permission to request user %s", requestor.Uuid, user.Uuid))
}

//get a project by id or return an error
func commonGetProject(project *model.MynahProject, requestor *model.MynahUser) error {
	//check that the user has permission to at least view this project
	if requestor.IsAdmin || project.GetPermissions(requestor) >= model.Read {
		return nil
	}
	return errors.New(fmt.Sprintf("user %s does not have permission to request project %s", requestor.Uuid, project.Uuid))
}

//check that the user is an admin (org checked in query)
func commonListUsers(requestor *model.MynahUser) error {
	if requestor.IsAdmin {
		return nil
	}
	return errors.New(fmt.Sprintf("user %s does not have permission to list users", requestor.Uuid))
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

//check that the creator is an admin
func commonCreateUser(user *model.MynahUser, creator *model.MynahUser) error {
	if !creator.IsAdmin {
		return errors.New(fmt.Sprintf("unable to create new user, user %s is not an admin", creator.Uuid))
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

//update a user in the database
func commonUpdateUser(user *model.MynahUser, requestor *model.MynahUser) error {
	if requestor.IsAdmin || requestor.Uuid == user.Uuid {
		return nil
	}
	return errors.New(fmt.Sprintf("user %s does not have permission to update user %s", requestor.Uuid, user.Uuid))
}

//update a project in the database
func commonUpdateProject(project *model.MynahProject, requestor *model.MynahUser) error {
	if requestor.IsAdmin || project.GetPermissions(requestor) >= model.Read {
		return nil
	}
	return errors.New(fmt.Sprintf("user %s does not have permission to update project %s", requestor.Uuid, project.Uuid))
}

//check that the requestor has permission
func commonDeleteUser(uuid *string, requestor *model.MynahUser) error {
	if requestor.IsAdmin {
		return nil
	}
	return errors.New(fmt.Sprintf("user %s does not have permission to update user %s", requestor.Uuid, *uuid))
}

//check that the requestor has permission to delete the project
func commonDeleteProject(project *model.MynahProject, requestor *model.MynahUser) error {
	if requestor.IsAdmin || project.GetPermissions(requestor) == model.Owner {
		return nil
	}
	return errors.New(fmt.Sprintf("user %s does not have permission to delete project %s", requestor.Uuid, project.Uuid))
}
