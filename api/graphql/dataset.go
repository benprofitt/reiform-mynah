// Copyright (c) 2022 by Reiform. All Rights Reserved.

package graphql

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/google/uuid"
	"github.com/graphql-go/graphql"
	"log"
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

//Handle new graphql requests for datasets
func DatasetQueryResolver(dbProvider db.DBProvider) (http.HandlerFunc, error) {
	//request datasets (query, list)
	var datasetQueryType = graphql.NewObject(
		graphql.ObjectConfig{
			Name: "DatasetQuery",
			Fields: graphql.Fields{
				//Get ?query={dataset(uuid:""){dataset_name}}
				"dataset": &graphql.Field{
					Type:        datasetType,
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
							return dbProvider.GetDataset(&uuid, user)
						} else {
							return nil, errors.New("graphql request missing uuid arg")
						}
					},
				},
				//Get list ?query={list{uuid,dataset_name}}
				"list": &graphql.Field{
					Type:        graphql.NewList(datasetType),
					Description: "Get dataset list",
					Resolve: func(p graphql.ResolveParams) (interface{}, error) {
						//get the authenticated user from context
						user := p.Context.Value(contextUserKey).(*model.MynahUser)
						//request datasets
						return dbProvider.ListDatasets(user)
					},
				},
			},
		})

	//update datasets (create, update)
	var datasetMutationType = graphql.NewObject(graphql.ObjectConfig{
		Name: "DatasetMutation",
		Fields: graphql.Fields{
			//Create a dataset ?query=mutation+_{create(dataset_name:"example name"){dataset_name}}
			"create": &graphql.Field{
				Type:        datasetType,
				Description: "Create a new dataset",
				Args: graphql.FieldConfigArgument{
					"dataset_name": &graphql.ArgumentConfig{
						Type: graphql.NewNonNull(graphql.String),
					},
				},
				Resolve: func(p graphql.ResolveParams) (interface{}, error) {
					user := p.Context.Value(contextUserKey).(*model.MynahUser)

					//create a new dataset
					dataset := model.MynahDataset{
						Uuid:            uuid.New().String(),
						ReferencedFiles: make([]string, 0),
						DatasetName:     p.Args["dataset_name"].(string),
					}

					//return the dataset and add to the database
					return dataset, dbProvider.CreateDataset(&dataset, user)
				},
			},

			//Update a dataset by uuid ?query=mutation+_{update(uuid:"",dataset_name:"update"){uuid,dataset_name}}
			"update": &graphql.Field{
				Type:        datasetType,
				Description: "Update a dataset by uuid",
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
					if dataset, err := dbProvider.GetDataset(&uuid, user); err == nil {
						//new name to use
						datasetName, datasetNameOk := p.Args["dataset_name"].(string)

						//update dataset
						if datasetNameOk {
							dataset.DatasetName = datasetName
						}

						//update the dataset
						return dataset, dbProvider.UpdateDataset(dataset, user)

					} else {
						return nil, err
					}
				},
			},
			//Delete a dataset by uuid ?query=mutation+_{delete(uuid:""){uuid,dataset_name}}
			"delete": &graphql.Field{
				Type:        datasetType,
				Description: "Delete dataset by uuid",
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
						return nil, dbProvider.DeleteDataset(&uuid, user)
					}
					return nil, errors.New("dataset delete request missing uuid")
				},
			},
		},
	})

	//create the schema
	schema, schemaErr := graphql.NewSchema(
		graphql.SchemaConfig{
			Query:    datasetQueryType,
			Mutation: datasetMutationType,
		},
	)

	if schemaErr != nil {
		log.Printf("failed to initialize graphql schema")
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
			log.Printf("graphql errors: %v", result.Errors)
			writer.WriteHeader(http.StatusBadRequest)
		} else {
			if err := json.NewEncoder(writer).Encode(result); err != nil {
				log.Printf("failed to write GQL response: %s", err)
			} else {
				writer.Header().Set("Content-Type", "application/json")
			}
		}
	}), nil
}
