// Copyright (c) 2022 by Reiform. All Rights Reserved.

package graphql

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/google/uuid"
	"github.com/graphql-go/graphql"
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/log"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

const contextUserKey = "user"

//Handle new graphql requests for projects
func ProjectQueryResolver(dbProvider db.DBProvider) (http.HandlerFunc, error) {
	//request projects (query, list)
	var projectQueryType = graphql.NewObject(
		graphql.ObjectConfig{
			Name: "ProjectQuery",
			Fields: graphql.Fields{
				//Get ?query={project(uuid:""){project_name}}
				"project": &graphql.Field{
					Type:        projectType,
					Description: "Get project by uuid",
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
							return dbProvider.GetProject(&uuid, user)
						} else {
							return nil, errors.New("graphql request missing uuid arg")
						}
					},
				},
				//Get list ?query={list{uuid,project_name}}
				"list": &graphql.Field{
					Type:        graphql.NewList(projectType),
					Description: "Get project list",
					Resolve: func(p graphql.ResolveParams) (interface{}, error) {
						//get the authenticated user from context
						user := p.Context.Value(contextUserKey).(*model.MynahUser)
						//request projects
						return dbProvider.ListProjects(user)
					},
				},
			},
		})

	//update projects (create, update)
	var projectMutationType = graphql.NewObject(graphql.ObjectConfig{
		Name: "ProjectMutation",
		Fields: graphql.Fields{
			//Create a project ?query=mutation+_{create(project_name:"example name"){project_name}}
			"create": &graphql.Field{
				Type:        projectType,
				Description: "Create a new project",
				Args: graphql.FieldConfigArgument{
					"project_name": &graphql.ArgumentConfig{
						Type: graphql.NewNonNull(graphql.String),
					},
				},
				Resolve: func(p graphql.ResolveParams) (interface{}, error) {
					user := p.Context.Value(contextUserKey).(*model.MynahUser)

					//create a new project
					project := model.MynahProject{
						Uuid:            uuid.New().String(),
						UserPermissions: make(map[string]model.ProjectPermissions),
						ProjectName:     p.Args["project_name"].(string),
					}

					//return the project and add to the database
					return project, dbProvider.CreateProject(&project, user)
				},
			},

			//Update a project by uuid ?query=mutation+_{update(uuid:"",project_name:"update"){uuid,project_name}}
			"update": &graphql.Field{
				Type:        projectType,
				Description: "Update a project by uuid",
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
					if project, err := dbProvider.GetProject(&uuid, user); err == nil {
						//new name to use
						projectName, projectNameOk := p.Args["project_name"].(string)

						//update project
						if projectNameOk {
							project.ProjectName = projectName
						}

						//update the project
						return project, dbProvider.UpdateProject(project, user)

					} else {
						return nil, err
					}
				},
			},
			//Delete a project by uuid ?query=mutation+_{delete(uuid:""){uuid,project_name}}
			"delete": &graphql.Field{
				Type:        projectType,
				Description: "Delete project by uuid",
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
						return nil, dbProvider.DeleteProject(&uuid, user)
					}
					return nil, errors.New("project delete request missing uuid")
				},
			},
		},
	})

	//create the schema
	schema, schemaErr := graphql.NewSchema(
		graphql.SchemaConfig{
			Query:    projectQueryType,
			Mutation: projectMutationType,
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
