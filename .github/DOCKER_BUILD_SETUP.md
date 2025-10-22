# GitHub Actions Docker Build & Push - Setup Guide

## Overview
This GitHub Actions workflow automatically builds and pushes Docker images to both Docker Hub and Google Cloud Artifact Registry (GCR) whenever code changes are pushed to the repository.

## Required GitHub Secrets

You need to configure the following secrets in your GitHub repository:

### 1. Docker Hub Credentials

#### `DOCKER_HUB_USERNAME`
Your Docker Hub username (e.g., `johnrcumming001`)

**How to get it:**
- This is your Docker Hub account username
- Find it at: https://hub.docker.com/settings/general

#### `DOCKER_HUB_TOKEN`
A Docker Hub access token (recommended over password)

**How to create it:**
1. Go to https://hub.docker.com/settings/security
2. Click "New Access Token"
3. Give it a description (e.g., "GitHub Actions DyLoRA-MoE")
4. Set permissions to "Read, Write, Delete"
5. Click "Generate"
6. **Copy the token immediately** (you won't be able to see it again)

### 2. Google Cloud Platform Credentials

#### `GCP_CREDENTIALS`
A JSON key for a GCP service account with permissions to push to Artifact Registry

**How to create it:**
1. Go to [GCP Console - Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Select your project (`dylora`)
3. Click "Create Service Account"
   - **Name:** `github-actions-docker-push`
   - **Description:** Service account for GitHub Actions to push Docker images
4. Click "Create and Continue"
5. Grant the following roles:
   - `Artifact Registry Writer` (or `roles/artifactregistry.writer`)
   - `Storage Object Viewer` (optional, for reading existing images)
6. Click "Continue" then "Done"
7. Click on the newly created service account
8. Go to the "Keys" tab
9. Click "Add Key" → "Create new key"
10. Choose "JSON" format
11. Click "Create" - the JSON key file will be downloaded
12. **Keep this file secure** - it contains sensitive credentials

**Alternative: Using existing service account**
If you already have a service account JSON key (like `dylora-ece7d9f1337d.json`), you can use that instead, but make sure it has the `Artifact Registry Writer` role:
```bash
gcloud projects add-iam-policy-binding dylora \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
  --role="roles/artifactregistry.writer"
```

## Adding Secrets to GitHub

### Via GitHub Web Interface:
1. Go to your repository: https://github.com/johnrcumming/DyLoRA-MoE
2. Click "Settings" tab
3. In the left sidebar, click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Add each secret:

   **For DOCKER_HUB_USERNAME:**
   - Name: `DOCKER_HUB_USERNAME`
   - Secret: `johnrcumming001` (or your username)
   
   **For DOCKER_HUB_TOKEN:**
   - Name: `DOCKER_HUB_TOKEN`
   - Secret: (paste the token from Docker Hub)
   
   **For GCP_CREDENTIALS:**
   - Name: `GCP_CREDENTIALS`
   - Secret: (paste the entire contents of the JSON key file)

### Via GitHub CLI:
```bash
# Set Docker Hub secrets
gh secret set DOCKER_HUB_USERNAME -b "johnrcumming001"
gh secret set DOCKER_HUB_TOKEN < docker_hub_token.txt

# Set GCP secret
gh secret set GCP_CREDENTIALS < /path/to/service-account-key.json
```

## Workflow Triggers

The workflow runs automatically on:

1. **Push to main or develop branches** when these files change:
   - `Dockerfile`
   - `requirements.txt`
   - `dylo_moe/**`
   - `data/**`
   - `benchmarks/**`
   - `train.py`
   - `entrypoint.py`
   - `build_and_push.py`
   - `.github/workflows/docker-build-push.yml`

2. **Pull Requests to main** (build only, no push)
   - Tests that the Docker image builds successfully
   - Does not push to any registry

3. **Manual trigger** (workflow_dispatch)
   - Go to Actions tab → "Build and Push Docker Images"
   - Click "Run workflow"
   - Choose whether to push or just build

## Workflow Behavior

### On Push to main/develop:
- ✅ Builds Docker image for linux/amd64
- ✅ Pushes to GCP Artifact Registry with tags: `{git-hash}`, `latest`
- ✅ Pushes to Docker Hub with tags: `{git-hash}`, `latest`

### On Pull Request:
- ✅ Builds Docker image (validation)
- ❌ Does NOT push to any registry

### Manual Workflow:
- ✅ Builds Docker image
- ✅/❌ Pushes based on `push_to_registries` input

## Pushed Image Locations

After a successful build and push:

### GCP Artifact Registry:
```
us-central1-docker.pkg.dev/dylora/dy-lora-training-repo/dy-lora-training-image:{git-hash}
us-central1-docker.pkg.dev/dylora/dy-lora-training-repo/dy-lora-training-image:latest
```

### Docker Hub:
```
johnrcumming001/dylora-moe:{git-hash}
johnrcumming001/dylora-moe:latest
```

## Monitoring

View workflow runs:
- https://github.com/johnrcumming/DyLoRA-MoE/actions

Each run shows:
- Build logs
- Push status
- Image tags created
- Summary with image URIs

## Troubleshooting

### "Error: Unable to authenticate to Docker Hub"
- Check that `DOCKER_HUB_USERNAME` and `DOCKER_HUB_TOKEN` secrets are set correctly
- Verify the token hasn't expired (Docker Hub tokens don't expire by default but can be revoked)
- Try regenerating the Docker Hub token

### "Error: Failed to authenticate to GCP"
- Verify `GCP_CREDENTIALS` secret contains valid JSON
- Check that the service account has `Artifact Registry Writer` role
- Ensure the service account key hasn't been deleted in GCP Console

### "Error: Permission denied to push to Artifact Registry"
```bash
# Fix: Add the role to your service account
gcloud projects add-iam-policy-binding dylora \
  --member="serviceAccount:SERVICE_ACCOUNT_EMAIL@dylora.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
```

### Build fails on pull requests
- This is expected if you're just testing the workflow setup
- The workflow only validates the build on PRs, not authentication

## Security Notes

⚠️ **Important Security Practices:**

1. **Never commit secrets** to the repository
2. **Use access tokens** instead of passwords for Docker Hub
3. **Rotate credentials** periodically (every 90 days recommended)
4. **Use minimal permissions** for service accounts
5. **Monitor secret usage** in GitHub Actions logs
6. **Revoke unused tokens** immediately

## Local Testing

To test the build locally before pushing:
```bash
# Build only
python build_and_push.py --build-only

# Build and push to Docker Hub only
python build_and_push.py --dockerhub-only

# Build and push to GCP only
python build_and_push.py --gcp-only

# Build and push to both (requires authentication)
python build_and_push.py
```
