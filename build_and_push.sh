#!/bin/bash
set -e

# Git Hash
GIT_HASH=$(git rev-parse --short HEAD)
export GIT_HASH

# Set your project and bucket details
PROJECT_ID="dylora"
REGION="us-central1"
DOCKER_REPO_NAME="dy-lora-training-repo"
IMAGE_NAME="dy-lora-training-image"

# Full image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${DOCKER_REPO_NAME}/${IMAGE_NAME}"

# Build the Docker image
echo "Building the Docker image for linux/amd64..."
docker buildx create --use || true # Create a new builder instance, or use the existing one
docker buildx build --platform linux/amd64 -t "${IMAGE_URI}:${GIT_HASH}" --push .
docker tag "${IMAGE_URI}:${GIT_HASH}" "${IMAGE_URI}:latest"

# Push the Docker image
echo "Pushing the Docker image to Artifact Registry..."
# The --push flag in the buildx command already pushes the image.
# We only need to push the 'latest' tag separately.
docker push "${IMAGE_URI}:latest"

echo "Docker image pushed successfully."