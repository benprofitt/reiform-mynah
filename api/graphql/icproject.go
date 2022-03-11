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

const contextUserKey = "user"

// ICProjectQueryResolver ProjectQueryResolver Handle new graphql requests for projects
func ICProjectQueryResolver(dbProvider db.DBProvider) (http.HandlerFunc, error) {
	//request projects (query, list)
	var icProjectQueryType = graphql.NewObject(
		graphql.ObjectConfig{
			Name: "ICProjectQuery",
			Fields: graphql.Fields{
				//Get ?query={project(uuid:""){project_name}}
				"icproject": &graphql.Field{
					Type:        icProjectType,
					Description: "Get project by uuid",
					Args: graphql.FieldConfigArgument{
						"uuid": &graphql.ArgumentConfig{
							Type: graphql.NewNonNull(graphql.String),
						},
					},
					Resolve: func(p graphql.ResolveParams) (interface{}, error) {
						if uuid, ok := p.Args["uuid"].(string); ok {
							//get the authenticated user from context
							user := p.Context.Value(contextUserKey).(*model.MynahUser)
							return dbProvider.GetICProject(&uuid, user)
						} else {
							return nil, errors.New("graphql request missing uuid arg")
						}
					},
				},
				//Get list ?query={list{uuid,project_name}}
				"list": &graphql.Field{
					Type:        graphql.NewList(icProjectType),
					Description: "Get project list",
					Resolve: func(p graphql.ResolveParams) (interface{}, error) {
						//get the authenticated user from context
						user := p.Context.Value(contextUserKey).(*model.MynahUser)
						//request projects
						return dbProvider.ListICProjects(user)
					},
				},
			},
		})

	//update projects (create, update)
	var icProjectMutationType = graphql.NewObject(graphql.ObjectConfig{
		Name: "ICProjectMutation",
		Fields: graphql.Fields{
			//Update a project by uuid ?query=mutation+_{update(uuid:"",project_name:"update"){uuid,project_name}}
			"update": &graphql.Field{
				Type:        icProjectType,
				Description: "Update an ic project by uuid",
				Args: graphql.FieldConfigArgument{
					"uuid": &graphql.ArgumentConfig{
						Type: graphql.NewNonNull(graphql.String),
					},
					"project_name": &graphql.ArgumentConfig{
						Type: graphql.String,
					},
				},
				Resolve: func(p graphql.ResolveParams) (interface{}, error) {
					//get the authenticated user from context to authorize db request
					user := p.Context.Value(contextUserKey).(*model.MynahUser)

					uuid, ok := p.Args["uuid"].(string)
					if !ok {
						return nil, errors.New("graphql update query missing project uuid")
					}

					//request the project
					if project, err := dbProvider.GetICProject(&uuid, user); err == nil {
						updatedFields := make([]string, 0)
						//update project
						if projectName, projectNameOk := p.Args["project_name"].(string); projectNameOk {
							updatedFields = append(updatedFields, "project_name")
							project.ProjectName = projectName
						}

						//update the project
						return project, dbProvider.UpdateICProject(project, user, updatedFields...)

					} else {
						return nil, err
					}
				},
			},
			//Delete a project by uuid ?query=mutation+_{delete(uuid:""){uuid}}
			"delete": &graphql.Field{
				Type:        icProjectType,
				Description: "Delete ic project by uuid",
				Args: graphql.FieldConfigArgument{
					"uuid": &graphql.ArgumentConfig{
						Type: graphql.NewNonNull(graphql.String),
					},
				},
				Resolve: func(p graphql.ResolveParams) (interface{}, error) {
					//get the authenticated user from context to authorize db request
					user := p.Context.Value(contextUserKey).(*model.MynahUser)
					if uuid, ok := p.Args["uuid"].(string); ok {
						return nil, dbProvider.DeleteICProject(&uuid, user)
					}
					return nil, errors.New("project delete request missing uuid")
				},
			},
		},
	})

	//create the schema
	schema, schemaErr := graphql.NewSchema(
		graphql.SchemaConfig{
			Query:    icProjectQueryType,
			Mutation: icProjectMutationType,
		},
	)

	if schemaErr != nil {
		return nil, schemaErr
	}

	//return a handler that executes queries
	return func(writer http.ResponseWriter, request *http.Request) {
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
	}, nil
}
