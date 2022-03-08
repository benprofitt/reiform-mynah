// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/graphql"
	"reiform.com/mynah/middleware"
)

//register the graphql endpoints
func registerGQLRoutes(router *middleware.MynahRouter, dbProvider db.DBProvider) error {
	projectHandler, projectErr := graphql.ProjectQueryResolver(dbProvider)
	if projectErr != nil {
		return projectErr
	}

	userHandler, userErr := graphql.UserQueryResolver(dbProvider)
	if userErr != nil {
		return userErr
	}

	datasetHandler, datasetErr := graphql.DatasetQueryResolver(dbProvider)
	if datasetErr != nil {
		return datasetErr
	}

	router.HandleHTTPRequest("POST", "graphql/project", projectHandler)
	router.HandleHTTPRequest("POST", "graphql/user", userHandler)
	router.HandleHTTPRequest("POST", "graphql/dataset", datasetHandler)

	return nil
}
