package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/settings"
	"syscall"
	"time"
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

	//initialize auth
	authProvider, authErr := auth.NewAuthProvider(settings)
	if authErr != nil {
		log.Fatalf("failed to initialize auth %s", authErr)
	}

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(settings, authProvider)
	if dbErr != nil {
		log.Fatalf("failed to initialize database connection %s", dbErr)
	}

	//initialize storage
	// storageProvider, storageErr := storage.NewStorageProvider(settings)
	// if storageErr != nil {
	// 	log.Fatalf("failed to initialize storage %s", storageErr)
	// }

	router := middleware.NewRouter(settings, authProvider, dbProvider)

	//run the server in a go routine
	go func() {
		router.ListenAndServe()
	}()

	//handle signals gracefully
	signalChan := make(chan os.Signal)
	signal.Notify(signalChan, os.Interrupt, syscall.SIGTERM)

	//block until signal received
	<-signalChan

	//shutdown the server (wait 15 seconds for any requests to finish)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*15)
	defer cancel()

	//close various services
	dbProvider.Close()
	authProvider.Close()
	router.Shutdown(ctx)
	os.Exit(0)
}
