#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: ${0} <image_tag>"
  exit 1
fi

IMAGE_TAG=$1
VOLUME_MOUNT="${PWD}/mynah_container_data_${IMAGE_TAG}/"
echo "NOTE: running mynah:${IMAGE_TAG}"
echo "NOTE: Writing container data to ${VOLUME_MOUNT}"

mkdir -p "${VOLUME_MOUNT}"

#Run detached with volume for data
docker run -d -p 8080:8080 \
  --volume "${VOLUME_MOUNT}:/mynah/data/" \
  "mynah:${IMAGE_TAG}"
