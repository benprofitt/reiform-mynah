package auth

import (
	"github.com/google/uuid"
	"reiform.com/mynah/model"
)

//create a new user
func (a *localAuth) CreateUser() (*model.MynahUser, string, error) {
	//TODO jwt
	return &model.MynahUser{
		Uuid:      uuid.New().String(),
		OrgId:     "",
		NameFirst: "",
		NameLast:  "",
		IsAdmin:   false,
	}, "", nil
}

//check the validity of the token
func (a *localAuth) IsValidToken(jwt *string) (string, error) {
	return "", nil
}
