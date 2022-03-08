// Copyright (c) 2022 by Reiform. All Rights Reserved.

package graphql

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/graphql-go/graphql"
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

// ICDatasetQueryResolver Handle new graphql requests for datasets
func ICDatasetQueryResolver(dbProvider db.DBProvider) (http.HandlerFunc, error) {
	//request datasets (query, list)
	var icDatasetQueryType = graphql.NewObject(
		graphql.ObjectConfig{
			Name: "ICDatasetQuery",
			Fields: graphql.Fields{
				//Get ?query={dataset(uuid:""){dataset_name}}
				"dataset": &graphql.Field{
					Type:        icDatasetType,
					Description: "Get dataset by uuid",
					Args: graphql.FieldConfigArgument{
						"uuid": &graphql.ArgumentConfig{
							Type: graphql.NewNonNull(graphql.String),
						},
					},
					Resolve: func(p graphql.ResolveParams) (interface{}, error) {
						uuid, ok := p.Args["uuid"].(string)
						if ok {
							//get the authenticated user from context
							user := p.Context.Value(contextUserKey).(*model.MynahUser)
							return dbProvider.GetICDataset(&uuid, user)
						} else {
							return nil, errors.New("graphql request missing uuid arg")
						}
					},
				},
				//Get list ?query={list{uuid,dataset_name}}
				"list": &graphql.Field{
					Type:        graphql.NewList(icDatasetType),
					Description: "Get ic dataset list",
					Resolve: func(p graphql.ResolveParams) (interface{}, error) {
						//get the authenticated user from context
						user := p.Context.Value(contextUserKey).(*model.MynahUser)
						//request datasets
						return dbProvider.ListICDatasets(user)
					},
				},
			},
		})

	//update datasets (create, update)
	var icDatasetMutationType = graphql.NewObject(graphql.ObjectConfig{
		Name: "ICDatasetMutation",
		Fields: graphql.Fields{
			//Update a dataset by uuid ?query=mutation+_{update(uuid:"",dataset_name:"update"){uuid,dataset_name}}
			"update": &graphql.Field{
				Type:        icDatasetType,
				Description: "Update an ic dataset by uuid",
				Args: graphql.FieldConfigArgument{
					"uuid": &graphql.ArgumentConfig{
						Type: graphql.NewNonNull(graphql.String),
					},
					"dataset_name": &graphql.ArgumentConfig{
						Type: graphql.String,
					},
				},
				Resolve: func(p graphql.ResolveParams) (interface{}, error) {
					//get the authenticated user from context to authorize db request
					user := p.Context.Value(contextUserKey).(*model.MynahUser)

					uuid, ok := p.Args["uuid"].(string)
					if !ok {
						return nil, errors.New("graphql update query missing dataset uuid")
					}

					//request the dataset
					if dataset, err := dbProvider.GetICDataset(&uuid, user); err == nil {
						//new name to use
						datasetName, datasetNameOk := p.Args["dataset_name"].(string)

						//update dataset
						if datasetNameOk {
							dataset.DatasetName = datasetName
						}

						//update the dataset
						return dataset, dbProvider.UpdateICDataset(dataset, user)

					} else {
						return nil, err
					}
				},
			},
			//Delete a dataset by uuid ?query=mutation+_{delete(uuid:""){uuid,dataset_name}}
			"delete": &graphql.Field{
				Type:        icDatasetType,
				Description: "Delete ic dataset by uuid",
				Args: graphql.FieldConfigArgument{
					"uuid": &graphql.ArgumentConfig{
						Type: graphql.NewNonNull(graphql.String),
					},
				},
				Resolve: func(p graphql.ResolveParams) (interface{}, error) {
					//get the authenticated user from context to authorize db request
					user := p.Context.Value(contextUserKey).(*model.MynahUser)
					uuid, ok := p.Args["uuid"].(string)
					if ok {
						return nil, dbProvider.DeleteICDataset(&uuid, user)
					}
					return nil, errors.New("dataset delete request missing uuid")
				},
			},
		},
	})

	//create the schema
	schema, schemaErr := graphql.NewSchema(
		graphql.SchemaConfig{
			Query:    icDatasetQueryType,
			Mutation: icDatasetMutationType,
		},
	)

	if schemaErr != nil {
		return nil, schemaErr
	}

	//return a handler that executes queries
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		//get the user from the request (used to authorize db requests)
		user := middleware.GetUserFromRequest(request)

		//execute the graphql query
		result := graphql.Do(graphql.Params{
			Schema:        schema,
			RequestString: request.URL.Query().Get("query"),
			Context:       context.WithValue(context.Background(), contextUserKey, user),
		})

		if len(result.Errors) > 0 {
			log.Warnf("graphql errors: %v", result.Errors)
			writer.WriteHeader(http.StatusBadRequest)
		} else {
			if err := json.NewEncoder(writer).Encode(result); err != nil {
				log.Errorf("failed to write GQL response: %s", err)
				writer.WriteHeader(http.StatusInternalServerError)
			} else {
				writer.Header().Set("Content-Type", "application/json")
			}
		}
	}), nil
}
