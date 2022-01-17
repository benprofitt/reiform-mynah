package api

import (
	"reiform.com/mynah/model"
)

//request type for an admin creating a user
type adminCreateUserRequest struct {
	//the first and last name to assign the user
	nameFirst string `json:"name_first"`
	nameLast  string `json:"name_last"`
}

//response type for an admin creating a user
type adminCreateUserResponse struct {
	//the generated jwt
	jwt string `json:"jwt"`
	//the user itself
	user model.MynahUser `json:"user"`
}
