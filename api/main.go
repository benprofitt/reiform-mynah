package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"reiform.com/mynah/api"
	"reiform.com/mynah/async"
	"reiform.com/mynah/auth"
	"reiform.com/mynah/db"
	"reiform.com/mynah/middleware"
	"reiform.com/mynah/python"
	"reiform.com/mynah/settings"
	"reiform.com/mynah/storage"
	"reiform.com/mynah/websockets"
	"syscall"
)

//entrypoint
func main() {
	log.SetPrefix("mynah ")

	settingsPtr := flag.String("settings", "mynah.json", "settings file path")
	flag.Parse()

	//load settings
	mynahSettings, settingsErr := settings.LoadSettings(settingsPtr)
	if settingsErr != nil {
		log.Fatalf("failed to load settings: %s", settingsErr)
	}

	//initialize auth
	authProvider, authErr := auth.NewAuthProvider(mynahSettings)
	if authErr != nil {
		log.Fatalf("failed to initialize auth %s", authErr)
	}

	//initialize the database connection
	dbProvider, dbErr := db.NewDBProvider(mynahSettings, authProvider)
	if dbErr != nil {
		log.Fatalf("failed to initialize database connection %s", dbErr)
	}

	//initialize storage
	storageProvider, storageErr := storage.NewStorageProvider(mynahSettings)
	if storageErr != nil {
		log.Fatalf("failed to initialize storage %s", storageErr)
	}

	//initialize python
	pythonProvider := python.NewPythonProvider(mynahSettings)

	//initialize websockets
	wsProvider := websockets.NewWebSocketProvider(mynahSettings)

	//initialize async workers
	asyncProvider := async.NewAsyncProvider(mynahSettings, wsProvider)

	//create the router and middleware
	router := middleware.NewRouter(mynahSettings, authProvider, dbProvider)

	//register api endpoints
	if err := api.RegisterRoutes(router,
		dbProvider,
		authProvider,
		storageProvider,
		pythonProvider,
		wsProvider,
		asyncProvider,
		mynahSettings); err != nil {
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

	//close various services
	router.Close()
	dbProvider.Close()
	authProvider.Close()
	pythonProvider.Close()
	storageProvider.Close()
	asyncProvider.Close()
	os.Exit(0)
}
