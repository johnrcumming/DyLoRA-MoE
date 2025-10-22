#!/bin/bash
# Quick setup script for GitHub secrets (requires GitHub CLI)
# Run this script to set up secrets for the Docker build workflow
# 
# Prerequisites:
# - GitHub CLI installed: https://cli.github.com/
# - Authenticated to GitHub: gh auth login
# - Docker Hub token ready
# - GCP service account JSON key ready

set -e

echo "🚀 DyLoRA-MoE Docker Build - GitHub Secrets Setup"
echo "=================================================="
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed"
    echo "   Install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated to GitHub"
    echo "   Run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI is ready"
echo ""

# Docker Hub Username
echo "📦 Docker Hub Configuration"
echo "---"
read -p "Enter Docker Hub username [johnrcumming001]: " DOCKER_USERNAME
DOCKER_USERNAME=${DOCKER_USERNAME:-johnrcumming001}
echo "Setting DOCKER_HUB_USERNAME..."
echo "$DOCKER_USERNAME" | gh secret set DOCKER_HUB_USERNAME
echo "✅ DOCKER_HUB_USERNAME set"
echo ""

# Docker Hub Token
echo "🔑 Docker Hub Token"
echo "   Create token at: https://hub.docker.com/settings/security"
echo "   Required permissions: Read, Write, Delete"
read -s -p "Paste Docker Hub token: " DOCKER_TOKEN
echo ""
if [ -z "$DOCKER_TOKEN" ]; then
    echo "❌ Docker Hub token cannot be empty"
    exit 1
fi
echo "$DOCKER_TOKEN" | gh secret set DOCKER_HUB_TOKEN
echo "✅ DOCKER_HUB_TOKEN set"
echo ""

# GCP Service Account
echo "☁️  Google Cloud Configuration"
echo "---"
echo "   Service account needs: Artifact Registry Writer role"
echo "   Create at: https://console.cloud.google.com/iam-admin/serviceaccounts"
read -p "Enter path to GCP service account JSON key: " GCP_KEY_PATH

if [ ! -f "$GCP_KEY_PATH" ]; then
    echo "❌ File not found: $GCP_KEY_PATH"
    exit 1
fi

# Validate JSON
if ! python3 -m json.tool "$GCP_KEY_PATH" > /dev/null 2>&1; then
    echo "❌ Invalid JSON file"
    exit 1
fi

echo "Setting GCP_CREDENTIALS..."
gh secret set GCP_CREDENTIALS < "$GCP_KEY_PATH"
echo "✅ GCP_CREDENTIALS set"
echo ""

# Summary
echo "=================================================="
echo "✅ All secrets configured successfully!"
echo ""
echo "Next steps:"
echo "1. Go to: https://github.com/johnrcumming/DyLoRA-MoE/settings/secrets/actions"
echo "2. Verify all three secrets are listed"
echo "3. Push a change to trigger the workflow"
echo ""
echo "Test the workflow:"
echo "  git add .github/workflows/docker-build-push.yml"
echo "  git commit -m 'Add Docker build workflow'"
echo "  git push"
echo ""
echo "Monitor at: https://github.com/johnrcumming/DyLoRA-MoE/actions"
echo "=================================================="
