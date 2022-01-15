package model

//interface that mynah database types adhere to
//standardizes the process of unique identification
type Identity interface {
	//get the unique identifier for this type
	GetUuid() string
	//get the org identifier for this type
	GetOrgId() string
}
