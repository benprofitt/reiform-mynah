package main

import (
	"flag"
	"log"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/settings"
	//"reiform.com/mynah/api"
	"reiform.com/mynah/middleware"
	//"reiform.com/mynah/storage"
)

//entrypoint
func main() {
	log.SetPrefix("mynah ")

	genPtr := flag.Bool("generate-settings", false, "just generate a settings file and then exit")
	settingsPtr := flag.String("settings", "mynah.json", "settings file path")
	flag.Parse()

	if *genPtr {
		settings.GenerateSettings(settingsPtr)
		log.Printf("generated settings file: %s", *settingsPtr)
	}

	//load settings
	settings, settingsErr := settings.LoadSettings(settingsPtr)
	if settingsErr != nil {
		log.Fatalf("failed to load settings: %s", settingsErr)
	}

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(settings)
	if dbErr != nil {
		log.Fatalf("failed to initialize database connection %s", dbErr)
	}

	//initialize auth
	authProvider, authErr := auth.NewAuthProvider(settings)
	if authErr != nil {
		log.Fatalf("failed to initialize auth %s", authErr)
	}

	// //initialize storage
	// storageProvider, storageErr := storage.NewStorageProvider(settings)
	// if storageErr != nil {
	// 	log.Fatalf("failed to initialize storage %s", storageErr)
	// }

	router := middleware.NewRouter(settings, authProvider, dbProvider)

	//listen and serve
	router.ListenAndServe()
}
