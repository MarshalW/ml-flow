#!/bin/bash

# Set the repository and tag for the image
REPOSITORY="ml-flow-ml-flow"
TAG="latest"

# Get the container ID of the running container based on the image
CONTAINER_ID=$(docker ps -a --filter "ancestor=${REPOSITORY}:${TAG}" -q | head -n 1)

if [ -z "$CONTAINER_ID" ]; then
  echo "Error: No container found based on ${REPOSITORY}:${TAG}"
  exit 1
fi

echo "Found container ID: $CONTAINER_ID"

# Commit the container changes to a new image with the same name and tag
docker commit "$CONTAINER_ID" "${REPOSITORY}:${TAG}"

if [ $? -eq 0 ]; then
  echo "Successfully updated image ${REPOSITORY}:${TAG}"
else
  echo "Error: Failed to commit changes to ${REPOSITORY}:${TAG}"
  exit 1
fi

# Optional: Remove dangling images to clean up
docker image prune -f

echo "Image update process completed"