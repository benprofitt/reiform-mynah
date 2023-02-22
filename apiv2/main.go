// Copyright (c) 2023 by Reiform. All Rights Reserved.

package main

import (
	"flag"
	"os"
	"os/signal"
	"reiform.com/mynah-api/api"
	"reiform.com/mynah-api/models/db"
	"reiform.com/mynah-api/services/log"
	"reiform.com/mynah-api/settings"
	"syscall"
)

func main() {
	log.Info("Reiform Mynah version %s", settings.MynahApplicationVersion)

	settingsPtr := flag.String("settings", "data/mynah.json", "settings file path")
	flag.Parse()

	//load settings
	err := settings.Load(settingsPtr)
	if err != nil {
		log.Fatal("failed to load settings: %s", err)
	}

	// initialize the database
	err = db.StartDBEnvironment()
	if err != nil {
		log.Fatal("failed to initialize database: %s", err)
	}

	mynahRouter := api.NewMynahRouter()

	//run the server in a go routine
	go func() {
		mynahRouter.ListenAndServe()
	}()

	//handle signals gracefully
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, os.Interrupt, syscall.SIGTERM)

	//block until signal received
	<-signalChan

	mynahRouter.Close()
	os.Exit(0)
}
