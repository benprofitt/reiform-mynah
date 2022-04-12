// Copyright (c) 2022 by Reiform. All Rights Reserved.

package log

import (
	"github.com/op/go-logging"
	"os"
)

var (
	log *logging.Logger
)

//initialize the logger
func init() {
	if log == nil {
		format := logging.MustStringFormatter(
			`mynah %{color}%{time:2006-01-02 15:04:05.999} %{level:.4s} %{color:reset} %{message}`,
		)
		//create a log backend
		backend := logging.NewLogBackend(os.Stderr, "", 0)
		backendFormat := logging.NewBackendFormatter(backend, format)

		//set the backend
		logging.SetBackend(backendFormat)

		log = logging.MustGetLogger("mynah")
	}
}

// Error log an error
func Error(args ...interface{}) {
	log.Error(args...)
}

// Errorf log a formatted error
func Errorf(format string, args ...interface{}) {
	log.Errorf(format, args...)
}

// Fatal log a fatal error
func Fatal(args ...interface{}) {
	log.Fatal(args...)
}

// Fatalf log a formatted fatal error
func Fatalf(format string, args ...interface{}) {
	log.Fatalf(format, args...)
}

// Info log info
func Info(args ...interface{}) {
	log.Info(args)
}

// Infof log formatted info
func Infof(format string, args ...interface{}) {
	log.Infof(format, args...)
}

// Warn log a warning
func Warn(args ...interface{}) {
	log.Warning(args...)
}

// Warnf log a formatted warning
func Warnf(format string, args ...interface{}) {
	log.Warningf(format, args...)
}
