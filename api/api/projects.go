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

		var req createICProjectRequest

		//attempt to parse the request body
		if err := requestParseJson(writer, request, &req); err != nil {
			log.Warnf("failed to parse json: %s", err)
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}

		//create the project, set the name and the files
		project, err := dbProvider.CreateICProject(user, func(project *model.MynahICProject) {
			project.ProjectName = req.Name
			project.Datasets = req.Datasets
		})

		if err != nil {
			log.Errorf("failed to add new ic project to database %s", err)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}

		//the response to return to the frontend
		res := createICProjectResponse{
			Project: *project,
		}

		//write the response
		if err := responseWriteJson(writer, &res); err != nil {
			log.Warnf("failed to write response as json: %s", err)
			writer.WriteHeader(http.StatusInternalServerError)
		}
	})
}
