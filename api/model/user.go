package model

//Defines a mynah user
type MynahUser struct {
	//the id of the user
	Uuid string `json:"-"`
	//the first name of the user
	NameFirst string `json:"name_first"`
	//the last name of the user
	NameLast string `json:"name_last"`
	//whether the user is an admin
	IsAdmin bool `json:"-"`
}
