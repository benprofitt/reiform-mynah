// Copyright (c) 2022 by Reiform. All Rights Reserved.

package middleware

import (
	"net/http"
	"reiform.com/mynah/log"
)

//Log requests for static resources
func (r *MynahRouter) staticLogger(handler http.Handler) http.HandlerFunc {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		res := logResponse{
			ResponseWriter: writer,
			status:         200,
		}

		//handle the request
		handler.ServeHTTP(&res, request)

		//log the result
		log.Infof("%s %s %s %d",
			request.Method,
			request.URL.Path,
			request.Proto,
			res.status)
	})
}

//serve the static site
func (r *MynahRouter) serveStaticSite() {
	log.Infof("serving static web resources from %s with path prefix %s",
		r.settings.StaticResourcesPath,
		r.settings.StaticPrefix)

	//serve static files
	fs := http.StripPrefix(r.settings.StaticPrefix,
		http.FileServer(http.Dir(r.settings.StaticResourcesPath)))
	r.PathPrefix(r.settings.StaticPrefix).Handler(r.staticLogger(fs))

	//redirect root
	r.Handle("/", http.RedirectHandler(r.settings.StaticPrefix, 301))
}
