package graphql

import (
	"github.com/graphql-go/graphql"
)

//Mynah Users (model.MynahUser)
var userType = graphql.NewObject(
	graphql.ObjectConfig{
		Name: "MynahUser",
		Fields: graphql.Fields{
			"uuid": &graphql.Field{
				Type: graphql.String,
			},
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

//Mynah Datasets (model.MynahDataset)
var datasetType = graphql.NewObject(
	graphql.ObjectConfig{
		Name: "MynahDataset",
		Fields: graphql.Fields{
			"uuid": &graphql.Field{
				Type: graphql.String,
			},
			"dataset_name": &graphql.Field{
				Type: graphql.String,
			},
			//TODO
			// "referenced_files": &graphql.Field{
			// 	Type: graphql.List,
			// },
		},
	},
)
