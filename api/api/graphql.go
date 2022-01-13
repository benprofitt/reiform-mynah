package api

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/graphql"
)

//register the graphql endpoints
func registerGQLRoutes(router *middleware.MynahRouter, dbProvider db.DBProvider) error {
  projectHandler, projectErr := graphql.ProjectQueryResolver(dbProvider)
  if projectErr != nil {
    return projectErr
  }

  //register the route
  router.HandleHTTPRequest("graphql/project", projectHandler)
}
