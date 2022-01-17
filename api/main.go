package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"reiform.com/mynah/api"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"
	"syscall"
	"time"
)

//entrypoint
func main() {
	log.SetPrefix("mynah ")

	settingsPtr := flag.String("settings", "mynah.json", "settings file path")
	flag.Parse()

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
	storageProvider, storageErr := storage.NewStorageProvider(settings)
	if storageErr != nil {
		log.Fatalf("failed to initialize storage %s", storageErr)
	}

	//initialize python
	pythonProvider := python.NewPythonProvider(settings)

	//initialize websockets
	wsProvider := websockets.NewWebSocketProvider(settings)

	//create the router and middleware
	router := middleware.NewRouter(settings, authProvider, dbProvider)

	//register api endpoints
	if err := api.RegisterRoutes(router,
		dbProvider,
		authProvider,
		storageProvider,
		pythonProvider,
		wsProvider,
		settings); err != nil {
		log.Fatalf("failed to initialize api routes: %s", err)
	}

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
	pythonProvider.Close()
	router.Shutdown(ctx)
	os.Exit(0)
}
