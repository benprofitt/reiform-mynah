// Copyright (c) 2023 by Reiform. All Rights Reserved.

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
			`mynah %{color}%{time:2006-01-02 15:04:05.999} ▶ %{level:.4s} %{color:reset} %{message}`,
		)
		//create a log backend
		backend := logging.NewLogBackend(os.Stderr, "", 0)
		backendFormat := logging.NewBackendFormatter(backend, format)

		//set the backend
		logging.SetBackend(backendFormat)

		log = logging.MustGetLogger("mynah")
	}
}

// Error log a formatted error
func Error(format string, args ...interface{}) {
	log.Errorf(format, args...)
}

// Fatal log a formatted fatal error
func Fatal(format string, args ...interface{}) {
	log.Fatalf(format, args...)
}

// Info log formatted info
func Info(format string, args ...interface{}) {
	log.Infof(format, args...)
}

// Warn log a formatted warning
func Warn(format string, args ...interface{}) {
	log.Warningf(format, args...)
}
