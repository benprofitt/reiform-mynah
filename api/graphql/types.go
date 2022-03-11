// Copyright (c) 2022 by Reiform. All Rights Reserved.

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

//Mynah Projects (model.MynahICProject)
var icProjectType = graphql.NewObject(
	graphql.ObjectConfig{
		Name: "MynahICProject",
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

//Mynah Image Classification Datasets (model.MynahICDataset)
var icDatasetType = graphql.NewObject(
	graphql.ObjectConfig{
		Name: "MynahICDataset",
		Fields: graphql.Fields{
			"uuid": &graphql.Field{
				Type: graphql.String,
			},
			"dataset_name": &graphql.Field{
				Type: graphql.String,
			},
		},
	},
)
