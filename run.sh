#!/bin/bash

mynah_api () {
  while true
  do
    ./mynah
    >&2 echo "ERROR: api server fell through"
  done
}

#start the frontend server
#TODO

#start the api server
mynah_api
