// Copyright (c) 2022 by Reiform. All Rights Reserved.

package middleware

import (
	"net/http"
	"path"
	"reiform.com/mynah/log"
)

//Log requests for static resources
func (r *MynahRouter) staticLogger(handler http.HandlerFunc) http.HandlerFunc {
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

	//redirect root -> static root index html
	r.Handle("/", http.RedirectHandler(path.Join(r.settings.StaticPrefix, "index.html"), 301))

	//create a file server that replaces the static directory with the mynah prefix
	fs := http.StripPrefix(r.settings.StaticPrefix,
		http.FileServer(http.Dir(r.settings.StaticResourcesPath)))

	//static react resources
	r.PathPrefix(path.Join(r.settings.StaticPrefix, "static/")).Handler(r.staticLogger(func(writer http.ResponseWriter, request *http.Request) {
		fs.ServeHTTP(writer, request)
	}))

	//react routes
	r.PathPrefix(r.settings.StaticPrefix).Handler(r.staticLogger(func(writer http.ResponseWriter, request *http.Request) {
		//serve the react index resource
		http.ServeFile(writer, request, path.Join(r.settings.StaticResourcesPath, "index.html"))
	}))
}
