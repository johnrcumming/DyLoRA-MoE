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

# Docker Hub configuration (set DOCKER_HUB_USERNAME env var or edit here)
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:-johnrcumming001}"
DOCKER_HUB_IMAGE="${DOCKER_HUB_USERNAME}/dylora-moe"

# Full image URIs
GCP_IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${DOCKER_REPO_NAME}/${IMAGE_NAME}"

# Build the Docker image with multiple tags
echo "Building the Docker image for linux/amd64..."
docker build --platform linux/amd64 \
  -t "${GCP_IMAGE_URI}:${GIT_HASH}" \
  -t "${GCP_IMAGE_URI}:latest" \
  -t "${DOCKER_HUB_IMAGE}:${GIT_HASH}" \
  -t "${DOCKER_HUB_IMAGE}:latest" \
  .

# Push to Google Cloud Artifact Registry
echo "Pushing the Docker image to GCP Artifact Registry..."
docker push "${GCP_IMAGE_URI}:${GIT_HASH}"
docker push "${GCP_IMAGE_URI}:latest"

# Push to Docker Hub
echo "Pushing the Docker image to Docker Hub..."
if docker push "${DOCKER_HUB_IMAGE}:${GIT_HASH}" && docker push "${DOCKER_HUB_IMAGE}:latest"; then
  echo "✓ Docker image pushed to Docker Hub successfully."
else
  echo "⚠ Warning: Failed to push to Docker Hub (may need to run 'docker login')"
  echo "  Continuing anyway - GCP push succeeded."
fi

echo ""
echo "================================"
echo "Docker image build complete!"
echo "================================"
echo "GCP Artifact Registry:"
echo "  - ${GCP_IMAGE_URI}:${GIT_HASH}"
echo "  - ${GCP_IMAGE_URI}:latest"
echo ""
echo "Docker Hub:"
echo "  - ${DOCKER_HUB_IMAGE}:${GIT_HASH}"
echo "  - ${DOCKER_HUB_IMAGE}:latest"
echo "================================"