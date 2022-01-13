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

	//register the route
	router.HandleHTTPRequest("graphql/project", projectHandler)
	router.HandleHTTPRequest("graphql/user", userHandler)

	return nil
}
