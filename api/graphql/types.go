package graphql

import (
	"github.com/graphql-go/graphql"
)

//Mynah Users (model.MynahUser)
var userType = graphql.NewObject(
	graphql.ObjectConfig{
		Name: "MynahUser",
		Fields: graphql.Fields{
			"name_first": &graphql.Field{
				Type: graphql.String,
			},
			"name_last": &graphql.Field{
				Type: graphql.String,
			},
		},
	},
)

//Mynah Projects (model.MynahProject)
var projectType = graphql.NewObject(
	graphql.ObjectConfig{
		Name: "MynahProject",
		Fields: graphql.Fields{
			"uuid": &graphql.Field{
				Type: graphql.String,
			},
			"project_name": &graphql.Field{
				Type: graphql.String,
			},
		},
	},
)
