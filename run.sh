#!/bin/bash

mynah_api () {
  while true
  do
    ./mynah
    >&2 echo "ERROR: api server fell through"
  done
}

mynah_api
