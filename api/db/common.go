// Copyright (c) 2022 by Reiform. All Rights Reserved.

package db

import (
	"errors"
	"fmt"
	"github.com/google/uuid"
	"reiform.com/mynah/log"
	"reiform.com/mynah/model"
	"time"
)

//check if any of the update keys are restricted
func restrictedKeys(keys []string) bool {
	if len(keys) == 0 {
		log.Warn("warning, database update has empty col key list")
	}
	for _, s := range keys {
		if (s == "org_id") || (s == "uuid") {
			return true
		}
	}
	return false
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
func commonGetProject(project model.MynahAbstractProject, requestor *model.MynahUser) error {
	//check that the user has permission to at least view this project
	if requestor.IsAdmin || project.GetBaseProject().GetPermissions(requestor) >= model.Read {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request project %s", requestor.Uuid, project.GetBaseProject().Uuid)
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
func commonGetDataset(dataset model.MynahAbstractDataset, requestor *model.MynahUser) error {
	//check that the user is the dataset owner (or admin)
	if requestor.IsAdmin || requestor.Uuid == dataset.GetBaseDataset().OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request dataset %s", requestor.Uuid, dataset.GetBaseDataset().Uuid)
}

//get a report
func commonGetReport(report model.MynahAbstractReport, requestor *model.MynahUser) error {
	//check that the user is the report owner (or admin)
	if requestor.IsAdmin || report.GetBaseReport().GetPermissions(requestor) >= model.Read {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to request report %s", requestor.Uuid, report.GetBaseReport().Uuid)
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

//get ic projects that the user can view
func commonListICProjects(projects []*model.MynahICProject, requestor *model.MynahUser) (filtered []*model.MynahICProject) {
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

//get the datasets that the user can view
func commonListICDatasets(datasets []*model.MynahICDataset, requestor *model.MynahUser) (filtered []*model.MynahICDataset) {
	//filter for files that this user has permission to view
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
	return &model.MynahUser{
		Uuid:      uuid.NewString(),
		OrgId:     creator.OrgId,
		NameFirst: "first",
		NameLast:  "last",
		IsAdmin:   false,
		CreatedBy: creator.Uuid,
	}, nil
}

//create a new project
func commonCreateProject(creator *model.MynahUser) *model.MynahProject {
	project := model.MynahProject{
		Uuid:            uuid.NewString(),
		OrgId:           creator.OrgId,
		UserPermissions: make(map[string]model.Permissions),
		ProjectName:     "name",
	}

	//give ownership permissions to the user
	project.UserPermissions[creator.Uuid] = model.Owner
	return &project
}

//create a new project
func commonCreateICProject(creator *model.MynahUser) *model.MynahICProject {
	project := model.MynahICProject{
		MynahProject: model.MynahProject{
			Uuid:            uuid.NewString(),
			OrgId:           creator.OrgId,
			UserPermissions: make(map[string]model.Permissions),
			ProjectName:     "name",
		},
		Datasets: make([]string, 0),
		Reports:  make([]string, 0),
	}
	//give ownership permissions to the user
	project.UserPermissions[creator.Uuid] = model.Owner
	return &project
}

//create a new file
func commonCreateFile(creator *model.MynahUser) *model.MynahFile {
	return &model.MynahFile{
		Uuid:                uuid.NewString(),
		OrgId:               creator.OrgId,
		OwnerUuid:           creator.Uuid,
		Name:                "file_name",
		Created:             time.Now().Unix(),
		DetectedContentType: "none",
		Versions:            make(map[model.MynahFileTag]*model.MynahFileVersion),
	}
}

//create a new dataset
func commonCreateDataset(creator *model.MynahUser) *model.MynahDataset {
	return &model.MynahDataset{
		Uuid:        uuid.NewString(),
		OrgId:       creator.OrgId,
		OwnerUuid:   creator.Uuid,
		DatasetName: "name",
	}
}

//create an ic dataset
func commonCreateICDataset(creator *model.MynahUser) *model.MynahICDataset {
	return &model.MynahICDataset{
		MynahDataset: model.MynahDataset{
			Uuid:        uuid.NewString(),
			OrgId:       creator.OrgId,
			OwnerUuid:   creator.Uuid,
			DatasetName: "name",
		},
		Files: make(map[string]*model.MynahICDatasetFile),
	}
}

//create a new ic diagnosis report
func commonCreateICDiagnosisReport(creator *model.MynahUser) *model.MynahICDiagnosisReport {
	report := model.MynahICDiagnosisReport{
		MynahReport: model.MynahReport{
			Uuid:            uuid.NewString(),
			OrgId:           creator.OrgId,
			UserPermissions: make(map[string]model.Permissions),
		},
		ImageIds:  make([]string, 0),
		ImageData: make(map[string]*model.MynahICDiagnosisReportImageMetadata),
		Breakdown: make(map[string]*model.MynahICDiagnosisReportBucket),
	}
	report.UserPermissions[creator.Uuid] = model.Owner
	return &report
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
func commonUpdateProject(project model.MynahAbstractProject, requestor *model.MynahUser, keys []string) error {
	//check that keys are not restricted
	if restrictedKeys(keys) {
		return errors.New("project update contained restricted keys")
	}

	if requestor.IsAdmin || project.GetBaseProject().GetPermissions(requestor) >= model.Read {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update project %s",
		requestor.Uuid, project.GetBaseProject().Uuid)
}

//update a dataset in the database
func commonUpdateDataset(dataset model.MynahAbstractDataset, requestor *model.MynahUser, keys []string) error {
	//check that keys are not restricted
	if restrictedKeys(keys) {
		return errors.New("dataset update contained restricted keys")
	}

	if requestor.IsAdmin || requestor.Uuid == dataset.GetBaseDataset().OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update dataset %s", requestor.Uuid, dataset.GetBaseDataset().Uuid)
}

//check that the requestor has permission
func commonDeleteUser(uuid *string, requestor *model.MynahUser) error {
	if requestor.IsAdmin {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to update user %s", requestor.Uuid, *uuid)
}

//check that the requestor has permission to delete the project
func commonDeleteProject(project model.MynahAbstractProject, requestor *model.MynahUser) error {
	if requestor.IsAdmin || project.GetBaseProject().GetPermissions(requestor) == model.Owner {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to delete project %s", requestor.Uuid, project.GetBaseProject().Uuid)
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
func commonDeleteDataset(dataset model.MynahAbstractDataset, requestor *model.MynahUser) error {
	if requestor.IsAdmin || requestor.Uuid == dataset.GetBaseDataset().OwnerUuid {
		return nil
	}
	return fmt.Errorf("user %s does not have permission to delete dataset %s", requestor.Uuid, dataset.GetBaseDataset().Uuid)
}
