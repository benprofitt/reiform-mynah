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
			`mynah %{color}%{time:2006-01-02T15:04:05.999Z} %{level:.4s} %{color:reset} %{message}`,
		)
		//create a log backend
		backend := logging.NewLogBackend(os.Stderr, "", 0)
		backendFormat := logging.NewBackendFormatter(backend, format)

		//set the backend
		logging.SetBackend(backendFormat)

		log = logging.MustGetLogger("mynah")
	}
}

//log an error
func Error(args ...interface{}) {
	log.Error(args...)
}

//log a formatted error
func Errorf(format string, args ...interface{}) {
	log.Errorf(format, args...)
}

//log a fatal error
func Fatal(args ...interface{}) {
	log.Fatal(args...)
}

//log a formatted fatal error
func Fatalf(format string, args ...interface{}) {
	log.Fatalf(format, args...)
}

//log info
func Info(args ...interface{}) {
	log.Info(args)
}

//log formatted info
func Infof(format string, args ...interface{}) {
	log.Infof(format, args...)
}

//log a warning
func Warn(args ...interface{}) {
	log.Warning(args...)
}

//log a formatted warning
func Warnf(format string, args ...interface{}) {
	log.Warningf(format, args...)
}
