package model

//Defines a mynah user
type MynahUser struct {
	//the id of the user
	Uuid string `json:"uuid"`
	//the id of the organization this user is part of
	OrgId string `json:"-"`
	//the first name of the user
	NameFirst string `json:"name_first"`
	//the last name of the user
	NameLast string `json:"name_last"`
	//whether the user is an admin
	IsAdmin bool `json:"-"`
	//who created this user
	CreatedBy string `json:"-"`
}

//get the user's uuid
func (u *MynahUser) GetUuid() string {
	return u.Uuid
}

//get the user's orgid
func (u *MynahUser) GetOrgId() string {
	return u.OrgId
}
