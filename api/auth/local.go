package auth

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"github.com/golang-jwt/jwt"
	"github.com/google/uuid"
	"log"
	"net/http"
	"os"
	"reiform.com/mynah/model"
	"reiform.com/mynah/settings"
)

//local auth client implements AuthProvider
type localAuth struct {
	//the jwt key loaded from file
	secret []byte
	//the header user to pass the jwt
	jwtHeader string
}

//check if the path to the key file exists
func privateKeyExists(path string) bool {
	if _, err := os.Stat(path); err == nil {
		return true
	} else if errors.Is(err, os.ErrNotExist) {
		return false
	} else {
		log.Fatalf("failed to identify whether private key exists: %s", err)
		return false
	}
}

//generate a new pem file to use for signing jwts
func generateJWTKeyFile(path string) error {
	//generate an RSA private key
	privatekey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return fmt.Errorf("failed to generate RSA key: %s", err)
	}

	//generate key bytes
	var pkBytes []byte = x509.MarshalPKCS1PrivateKey(privatekey)
	pkBlock := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: pkBytes,
	}

	//write the private key to a file
	pemFile, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create private key %s: %s", path, err)
	}
	defer pemFile.Close()

	//write the data to the file
	err = pem.Encode(pemFile, pkBlock)
	if err != nil {
		return fmt.Errorf("failed to encode private key %s: %s", path, err)
	}
	return nil
}

//initialize the local auth provider
func newLocalAuth(mynahSettings *settings.MynahSettings) (*localAuth, error) {
	//check if the key file exists
	if !privateKeyExists(mynahSettings.AuthSettings.PemFilePath) {
		//generate a new key file
		if err := generateJWTKeyFile(mynahSettings.AuthSettings.PemFilePath); err != nil {
			return nil, err
		}
		log.Printf("generated jwt signing key %s", mynahSettings.AuthSettings.PemFilePath)
	}

	//load the jwt signing key
	if data, fileErr := os.ReadFile(mynahSettings.AuthSettings.PemFilePath); fileErr == nil {
		//create the local auth provider with the loaded secret
		return &localAuth{
			secret:    data,
			jwtHeader: mynahSettings.AuthSettings.JwtHeader,
		}, nil
	} else {
		return nil, fileErr
	}
}

//create a signed jwt
func (a *localAuth) createSignedJWT(uuid string) (string, error) {
	//create a new token
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"uuid": uuid,
	})
	return token.SignedString(a.secret)
}

//create a new user
func (a *localAuth) CreateUser() (*model.MynahUser, string, error) {
	//create a new uuid
	uuid := uuid.New().String()
	//create a jwt and a new user
	if jwt, err := a.createSignedJWT(uuid); err == nil {
		return &model.MynahUser{
			Uuid:    uuid,
			IsAdmin: false,
		}, jwt, nil
	} else {
		return nil, "", err
	}
}

//check the validity of the jwt in the header, if valid, return user uuid
func (a *localAuth) IsAuthReq(req *http.Request) (string, error) {
	//get the jwt from the header
	jwtToken := req.Header.Get(a.jwtHeader)
	if jwtToken == "" {
		return "", fmt.Errorf("request missing header %s", a.jwtHeader)
	}

	//parse the token
	token, err := jwt.Parse(jwtToken, func(token *jwt.Token) (interface{}, error) {
		//check for HMAC algorithm
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return a.secret, nil
	})

	if err != nil {
		return "", fmt.Errorf("error decoding jwt %s", err)
	}

	//get the claims
	if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
		return fmt.Sprintf("%v", claims["uuid"]), nil

	} else {
		return "", errors.New("failed to decode jwt")
	}
}

//close the auth provider
func (a *localAuth) Close() {
	log.Printf("local authentication shutdown")
}
