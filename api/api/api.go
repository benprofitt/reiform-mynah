package api

import (
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/settings"
)

//Register all api routes
func RegisterRoutes(router *middleware.MynahRouter, dbProvider db.DBProvider, settings *settings.MynahSettings) {
	//TODO
}
