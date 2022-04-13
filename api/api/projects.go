// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

// icProjectCreate creates a new project in the database
func icProjectCreate(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request (will be the owner)
		user := middleware.GetUserFromRequest(request)

		var req CreateICProjectRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//create the project, set the name and the files
		project, err := dbProvider.CreateICProject(user, func(project *model.MynahICProject) error {
			project.ProjectName = req.Name
			project.Datasets = req.Datasets
			return nil
		})

		if err != nil {
			log.Errorf("failed to add new ic project to database %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//write the response
		if err := responseWriteJson(writer, &project); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

// odProjectCreate creates a new project in the database
func odProjectCreate(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request (will be the owner)
		user := middleware.GetUserFromRequest(request)

		var req CreateODProjectRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//create the project, set the name and the files
		project, err := dbProvider.CreateODProject(user, func(project *model.MynahODProject) error {
			project.ProjectName = req.Name
			project.Datasets = req.Datasets
			return nil
		})

		if err != nil {
			log.Errorf("failed to add new od project to database %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//write the response
		if err := responseWriteJson(writer, &project); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

// icProjectList lists ic projects
func icProjectList(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		//list all ic projects
		projects, err := dbProvider.ListICProjects(user)

		if err != nil {
			log.Errorf("failed to list ic projects in database %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//write the response
		if err := responseWriteJson(writer, &projects); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}

// odProjectList lists ic projects
func odProjectList(dbProvider db.DBProvider) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//the user making the request
		user := middleware.GetUserFromRequest(request)

		//list all ic projects
		projects, err := dbProvider.ListODProjects(user)

		if err != nil {
			log.Errorf("failed to list od projects in database %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}

		//write the response
		if err := responseWriteJson(writer, &projects); err != nil {
			log.Errorf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
