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

// UserQueryResolver Handle new graphql requests for users
func UserQueryResolver(dbProvider db.DBProvider) (http.HandlerFunc, error) {
	//request users (query, list)
	var userQueryType = graphql.NewObject(
		graphql.ObjectConfig{
			Name: "UserQuery",
			Fields: graphql.Fields{
				//Get ?query={user(uuid:""){first_name,last_name}}
				"user": &graphql.Field{
					Type:        userType,
					Description: "Get user by uuid",
					Args: graphql.FieldConfigArgument{
						"uuid": &graphql.ArgumentConfig{
							Type: graphql.NewNonNull(graphql.String),
						},
					},
					Resolve: func(p graphql.ResolveParams) (interface{}, error) {
						if uuid, ok := p.Args["uuid"].(string); ok {
							//get the authenticated user from context
							user := p.Context.Value(contextUserKey).(*model.MynahUser)
							return dbProvider.GetUser(&uuid, user)
						} else {
							return nil, errors.New("graphql request missing uuid arg")
						}
					},
				},
				//Get list ?query={list{uuid,last_name,first_name}}
				"list": &graphql.Field{
					Type:        graphql.NewList(userType),
					Description: "Get user list",
					Resolve: func(p graphql.ResolveParams) (interface{}, error) {
						//get the authenticated user from context
						user := p.Context.Value(contextUserKey).(*model.MynahUser)
						//request users
						return dbProvider.ListUsers(user)
					},
				},
			},
		})

	//update users (create, update)
	var userMutationType = graphql.NewObject(graphql.ObjectConfig{
		Name: "UserMutation",
		Fields: graphql.Fields{
			//Update a user by uuid ?query=mutation+_{update(uuid:"",name_first:"update",name_last:"name last"){uuid,name_first,name_last}}
			"update": &graphql.Field{
				Type:        userType,
				Description: "Update a user by uuid",
				Args: graphql.FieldConfigArgument{
					"uuid": &graphql.ArgumentConfig{
						Type: graphql.NewNonNull(graphql.String),
					},
					"name_first": &graphql.ArgumentConfig{
						Type: graphql.String,
					},
					"name_last": &graphql.ArgumentConfig{
						Type: graphql.String,
					},
				},
				Resolve: func(p graphql.ResolveParams) (interface{}, error) {
					//get the authenticated user from context to authorize db request
					user := p.Context.Value(contextUserKey).(*model.MynahUser)

					uuid, ok := p.Args["uuid"].(string)
					if !ok {
						return nil, errors.New("graphql update query missing user uuid")
					}

					//request the user
					if updateUser, err := dbProvider.GetUser(&uuid, user); err == nil {
						updatedFields := make([]string, 0)
						//update user
						if nameFirst, nameFirstOk := p.Args["name_first"].(string); nameFirstOk {
							updatedFields = append(updatedFields, "name_first")
							updateUser.NameFirst = nameFirst
						}
						if nameLast, nameLastOk := p.Args["name_last"].(string); nameLastOk {
							updatedFields = append(updatedFields, "name_last")
							updateUser.NameLast = nameLast
						}

						//update the user
						return updateUser, dbProvider.UpdateUser(updateUser, user, updatedFields...)

					} else {
						return nil, err
					}
				},
			},
			//Delete a user by uuid ?query=mutation+_{delete(uuid:""){uuid}}
			"delete": &graphql.Field{
				Type:        userType,
				Description: "Delete user by uuid",
				Args: graphql.FieldConfigArgument{
					"uuid": &graphql.ArgumentConfig{
						Type: graphql.NewNonNull(graphql.String),
					},
				},
				Resolve: func(p graphql.ResolveParams) (interface{}, error) {
					//get the authenticated user from context to authorize db request
					user := p.Context.Value(contextUserKey).(*model.MynahUser)
					if uuid, ok := p.Args["uuid"].(string); ok {
						return nil, dbProvider.DeleteUser(&uuid, user)
					}
					return nil, errors.New("user delete request missing uuid")
				},
			},
		},
	})

	//create the schema
	schema, schemaErr := graphql.NewSchema(
		graphql.SchemaConfig{
			Query:    userQueryType,
			Mutation: userMutationType,
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
