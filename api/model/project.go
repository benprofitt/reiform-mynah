package model

//the permissions a user can have for a project
type ProjectPermissions int

const (
	None ProjectPermissions = iota
	Read
	Edit
	Owner
)

//Defines a mynah project
type MynahProject struct {
	//the id of the project
	Uuid string `json:"uuid"`
	//the id of the organization this project is part of
	OrgId string `json:"-"`
	//permissions that various users have
	UserPermissions map[string]ProjectPermissions `json:"-"`
	//the name of the project
	ProjectName string `json:"project_name"`
}

//Get the permissions that a user has on a given project
func (p *MynahProject) GetPermissions(user *MynahUser) ProjectPermissions {
	if v, found := p.UserPermissions[user.Uuid]; found {
		return v
	} else {
		return None
	}
}
