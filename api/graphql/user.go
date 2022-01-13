package graphql

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/graphql-go/graphql"
	"log"
	"net/http"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/model"
)

//Handle new graphql requests for users
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
						uuid, ok := p.Args["uuid"].(string)
						if ok {
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

					//new name to use
					nameFirst, nameFirstOk := p.Args["name_first"].(string)
					nameLast, nameLastOk := p.Args["name_last"].(string)

					//request the user
					if updateUser, err := dbProvider.GetUser(&uuid, user); err == nil {
						//update user
						if nameFirstOk {
							updateUser.NameFirst = nameFirst
						}
						if nameLastOk {
							updateUser.NameLast = nameLast
						}

						//update the user
						return updateUser, dbProvider.UpdateUser(updateUser, user)

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
					uuid, ok := p.Args["uuid"].(string)
					if ok {
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
			json.NewEncoder(writer).Encode(result)
		}
	}), nil
}
