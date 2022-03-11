// Copyright (c) 2022 by Reiform. All Rights Reserved.

package api

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/graphql"
	"reiform.com/mynah/middleware"
)

//register the graphql endpoints
func registerGQLRoutes(router *middleware.MynahRouter, dbProvider db.DBProvider) error {
	userHandler, userErr := graphql.UserQueryResolver(dbProvider)
	if userErr != nil {
		return userErr
	}

	//projectHandler, projectErr := graphql.ICProjectQueryResolver(dbProvider)
	//if projectErr != nil {
	//	return projectErr
	//}
	//
	//datasetHandler, datasetErr := graphql.ICDatasetQueryResolver(dbProvider)
	//if datasetErr != nil {
	//	return datasetErr
	//}

	router.HandleHTTPRequest("GET", "graphql/user", userHandler)

	// FIXME: known issue for anonymous struct composition: https://github.com/graphql-go/graphql/issues/553
	//router.HandleHTTPRequest("graphql/icproject", projectHandler)
	//router.HandleHTTPRequest("graphql/icdataset", datasetHandler)

	return nil
}
