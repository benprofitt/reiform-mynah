package api

import (
	"reiform.com/mynah/model"
)

//request type for an admin creating a user
type adminCreateUserRequest struct {
	//the first and last name to assign the user
	NameFirst string `json:"name_first"`
	NameLast  string `json:"name_last"`
}

//response type for an admin creating a user
type adminCreateUserResponse struct {
	//the generated jwt
	Jwt string `json:"jwt"`
	//the user itself
	User model.MynahUser `json:"user"`
}
