package auth

//Defines the interface the auth client must implement
type AuthProvider interface {
	//Create a new user, returns the user's uuid, jwt if applicable
	CreateUser() (string, string)
	//Takes a JWT token attached to a request and verifies that the token is
	//valid. If valid, returns the user's uuid. If invalid, returns an error
	IsValidToken(*string) (string, error)
}

//local auth client adheres to AuthProvider
type localAuth struct {
}

//external auth client adheres to AuthProvider
type externalAuth struct {
}
