// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"fmt"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"time"
)

//check if any of the update keys are restricted
func restrictedKeys(keys MynahDBColumns) error {
	if keys.empty() {
		log.Warn("warning, database update has empty col key list")
	}

	if ok, restrictedCol := keys.containsRestricted(); ok {
		return fmt.Errorf("update contains restricted column: %s", restrictedCol)
	}
	return nil
}

var orgIdCondition = fmt.Sprintf("%s = ?", model.OrgIdColName)

//verify that the requestor is an admin
func commonGetUser(user *model.MynahUser, requestor *model.MynahUser) error {
	//Note: this is the only location where we need to compare org id (it isn't filtered in the query
	//so that auth works with no knowledge of org)
	if (user.OrgId == requestor.OrgId) && requestor.IsAdmin {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request user %s", requestor.Uuid, user.Uuid)
}

//get a file by id or return an error
func commonGetFile(file *model.MynahFile, requestor *model.MynahUser) error {
	//check that the user is the file owner (or admin)
	if requestor.IsAdmin || file.GetPermissions(requestor) >= model.Read {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request file %s", requestor.Uuid, file.Uuid)
}

//get a dataset by id or return an error
func commonGetDataset(dataset model.MynahAbstractDataset, requestor *model.MynahUser) error {
	//check that the user is the dataset owner (or admin)
	if requestor.IsAdmin || dataset.GetBaseDataset().GetPermissions(requestor) >= model.Read {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request dataset %s", requestor.Uuid, dataset.GetBaseDataset().Uuid)
}

//check that the user is an admin (org checked in query)
func commonListUsers(requestor *model.MynahUser) error {
	if requestor.IsAdmin {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to list users", requestor.Uuid)
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
func commonListICDatasets(datasets []*model.MynahICDataset, requestor *model.MynahUser) (filtered []*model.MynahICDataset) {
	//filter for datasets that this user has permission to view
	for _, d := range datasets {
		if e := commonGetDataset(d, requestor); e == nil {
			filtered = append(filtered, d)
		}
	}
	return filtered
}

//get the datasets that the user can view
func commonListODDatasets(datasets []*model.MynahODDataset, requestor *model.MynahUser) (filtered []*model.MynahODDataset) {
	//filter for datasets that this user has permission to view
	for _, d := range datasets {
		if e := commonGetDataset(d, requestor); e == nil {
			filtered = append(filtered, d)
		}
	}
	return filtered
}

//create a user
func commonCreateUser(creator *model.MynahUser) (*model.MynahUser, error) {
	if !creator.IsAdmin {
		return nil, fmt.Errorf("only admins can create users, %s is not an admin", creator.Uuid)
	}
	return model.NewUser(creator), nil
}

//update a user in the database
func commonUpdateUser(user *model.MynahUser, requestor *model.MynahUser, keys MynahDBColumns) error {
	//check that keys are not restricted
	if err := restrictedKeys(keys); err != nil {
		return fmt.Errorf("user update failed: %s", err)
	}

	if requestor.IsAdmin || requestor.Uuid == user.Uuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update user %s", requestor.Uuid, user.Uuid)
}

//update a dataset in the database
func commonUpdateDataset(dataset model.MynahAbstractDataset, requestor *model.MynahUser, keys MynahDBColumns) error {
	//check that keys are not restricted
	if err := restrictedKeys(keys); err != nil {
		return fmt.Errorf("dataset update failed: %s", err)
	}
	dataset.GetBaseDataset().DateModified = time.Now().Unix()
	keys.add(model.DateModifiedCol)

	if requestor.IsAdmin || dataset.GetBaseDataset().GetPermissions(requestor) >= model.Edit {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update dataset %s", requestor.Uuid, dataset.GetBaseDataset().Uuid)
}

//update a file metadata
func commonUpdateFile(file *model.MynahFile, requestor *model.MynahUser, keys MynahDBColumns) error {
	//check that keys are not restricted
	if err := restrictedKeys(keys); err != nil {
		return fmt.Errorf("file update failed: %s", err)
	}
	if requestor.IsAdmin || file.GetPermissions(requestor) >= model.Edit {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update file %s", requestor.Uuid, file.Uuid)
}

//update a file metadata
func commonUpdateFiles(files model.MynahFileSet, requestor *model.MynahUser, keys MynahDBColumns) error {
	//check that keys are not restricted
	if err := restrictedKeys(keys); err != nil {
		return fmt.Errorf("file update failed: %s", err)
	}
	for _, file := range files {
		if !requestor.IsAdmin && file.GetPermissions(requestor) < model.Edit {
			return fmt.Errorf("user %s does not have permission to update file %s", requestor.Uuid, file.Uuid)
		}
	}
	return nil
}

//check that the requestor has permission
func commonDeleteUser(uuid model.MynahUuid, requestor *model.MynahUser) error {
	if requestor.IsAdmin {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update user %s", requestor.Uuid, uuid)
}

//check that the requestor has permission to delete the file
func commonDeleteFile(file *model.MynahFile, requestor *model.MynahUser) error {
	//TODO check if file is part of dataset
	if requestor.IsAdmin || file.GetPermissions(requestor) == model.Owner {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to delete file %s", requestor.Uuid, file.Uuid)
}

//check that the requestor has permission to delete the dataset
func commonDeleteDataset(dataset model.MynahAbstractDataset, requestor *model.MynahUser) error {
	if requestor.IsAdmin || dataset.GetBaseDataset().GetPermissions(requestor) == model.Owner {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to delete dataset %s", requestor.Uuid, dataset.GetBaseDataset().Uuid)
}
