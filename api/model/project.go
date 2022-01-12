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
	Uuid string `json:"-"`
	//the id of the organization this project is part of
	OrgId string `json:"-"`
	//permissions that various users have
	userPermissions map[string]ProjectPermissions `json:"-"`
	//the name of the project
	ProjectName string `json:"project_name"`
}

//Get the permissions that a user has on a given project
func (p *MynahProject) GetPermissions(user *MynahUser) ProjectPermissions {
	if v, found := p.userPermissions[user.Uuid]; found {
		return v
	} else {
		return None
	}
}

//Add permissions for a user
func (p *MynahProject) AddPermissions(user *MynahUser, perm ProjectPermissions) {
	p.userPermissions[user.Uuid] = perm
}
