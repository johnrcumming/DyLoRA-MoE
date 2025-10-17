# Docker Setup Guide

## Overview

The DyLoRA-MoE Docker image is now pushed to **two registries**:
1. **Google Cloud Artifact Registry** (for Vertex AI training)
2. **Docker Hub** (for public access and local development)

---

## Prerequisites

### 1. Docker Hub Account

1. Create account at https://hub.docker.com
2. Login locally:
   ```bash
   docker login
   # Enter username and password
   ```

### 2. Google Cloud Authentication

```bash
# Configure Docker to use gcloud credentials
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## Building and Pushing Images

### Standard Build (Both Registries)

```bash
./build_and_push.sh
```

This will:
- ✅ Build Docker image for `linux/amd64`
- ✅ Tag with Git hash and `latest`
- ✅ Push to GCP Artifact Registry (`us-central1-docker.pkg.dev/dylora/...`)
- ✅ Push to Docker Hub (`johnrcumming/dylora-moe:latest`)

### Custom Docker Hub Username

```bash
# Set environment variable before running
export DOCKER_HUB_USERNAME=yourusername
./build_and_push.sh
```

Or edit `build_and_push.sh` line 15:
```bash
DOCKER_HUB_USERNAME="${DOCKER_HUB_USERNAME:-yourusername}"
```

---

## Image Locations

### Google Cloud Artifact Registry (Private)
```
us-central1-docker.pkg.dev/dylora/dy-lora-training-repo/dy-lora-training-image:latest
us-central1-docker.pkg.dev/dylora/dy-lora-training-repo/dy-lora-training-image:<git-hash>
```

**Used for:** Vertex AI training jobs

### Docker Hub (Public)
```
johnrcumming/dylora-moe:latest
johnrcumming/dylora-moe:<git-hash>
```

**Used for:** 
- Public distribution
- Local development
- CI/CD pipelines
- Community contributions

---

## Pulling Images

### From Docker Hub (Public - No Auth Required)

```bash
# Latest version
docker pull johnrcumming/dylora-moe:latest

# Specific version
docker pull johnrcumming/dylora-moe:abc1234
```

### From GCP Artifact Registry (Requires GCP Auth)

```bash
# Configure authentication first
gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull image
docker pull us-central1-docker.pkg.dev/dylora/dy-lora-training-repo/dy-lora-training-image:latest
```

---

## Running Containers Locally

### Using Docker Hub Image

```bash
# Interactive shell
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  johnrcumming/dylora-moe:latest \
  bash

# Run training directly
docker run --rm \
  --gpus all \
  -v $(pwd):/workspace \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  johnrcumming/dylora-moe:latest \
  python train.py --bf16 --num_epochs 3 --training_subset 10
```

---

## Troubleshooting

### Docker Hub Push Fails

**Error:** `denied: requested access to the resource is denied`

**Solution:**
```bash
# Login to Docker Hub
docker login
# Enter your Docker Hub username and password

# Retry build
./build_and_push.sh
```

### GCP Artifact Registry Push Fails

**Error:** `unauthorized: You don't have the needed permissions`

**Solution:**
```bash
# Authenticate with GCP
gcloud auth login

# Configure Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Retry build
./build_and_push.sh
```

### Build Fails: "platform mismatch"

**Error:** Building on Apple Silicon (M1/M2) but need `linux/amd64`

**Solution:** Script already includes `--platform linux/amd64` flag. Ensure Docker Desktop is updated.

---

## Image Versioning Strategy

### Tags

1. **`:latest`** - Most recent build (auto-updated on each push)
2. **`:<git-hash>`** - Specific commit (e.g., `:abc1234`)
3. **`:v1.0.0`** (future) - Semantic versioning for releases

### When to Use Each Tag

| Use Case | Recommended Tag |
|----------|----------------|
| **Vertex AI Training** | `:latest` (always uses newest code) |
| **Local Development** | `:latest` (iterate quickly) |
| **Production Deployment** | `:<git-hash>` or `:v1.0.0` (reproducible) |
| **CI/CD Testing** | `:<git-hash>` (exact version tested) |
| **Paper Experiments** | `:<git-hash>` (reproducibility) |

---

## GitHub Actions Integration (Future)

To automate Docker Hub pushes on every commit:

```yaml
# .github/workflows/docker-publish.yml
name: Docker Build and Push

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          platforms: linux/amd64
          push: true
          tags: |
            johnrcumming/dylora-moe:latest
            johnrcumming/dylora-moe:${{ github.sha }}
```

---

## Docker Hub Repository Settings

### Recommended Configuration

1. **Visibility:** Public (for community access)
2. **Description:** 
   ```
   DyLoRA-MoE: Dynamic LoRA-based Mixture-of-Experts for efficient code generation training
   ```
3. **README:** Link to GitHub repository
4. **Automated Builds:** Connect to GitHub repo (optional)

### Creating Docker Hub Repository

1. Go to https://hub.docker.com
2. Click "Create Repository"
3. Name: `dylora-moe`
4. Visibility: Public
5. Click "Create"

---

## Image Size Optimization (Future)

Current image size: ~15GB (includes PyTorch, CUDA, HuggingFace transformers)

**Optimization strategies:**
- Multi-stage builds (separate build and runtime)
- Remove unnecessary CUDA components
- Use smaller base image (e.g., `python:3.11-slim` + manual CUDA install)
- Compress layers with `docker squash`

**Trade-off:** Smaller images vs. build time and compatibility

---

## Security Best Practices

1. **Never commit credentials:**
   - ❌ Don't hardcode `HF_TOKEN`, `WANDB_API_KEY` in Dockerfile
   - ✅ Pass as environment variables at runtime

2. **Use secrets for CI/CD:**
   - Store Docker Hub token as GitHub secret
   - Rotate tokens regularly

3. **Scan images for vulnerabilities:**
   ```bash
   docker scan johnrcumming/dylora-moe:latest
   ```

4. **Keep base images updated:**
   ```dockerfile
   # Regularly update base image in Dockerfile
   FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
   ```

---

## Quick Reference

### Build and Push to Both Registries
```bash
./build_and_push.sh
```

### Pull from Docker Hub
```bash
docker pull johnrcumming/dylora-moe:latest
```

### Run Locally
```bash
docker run -it --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  johnrcumming/dylora-moe:latest bash
```

### Check Image Info
```bash
docker images | grep dylora
docker inspect johnrcumming/dylora-moe:latest
```

---

## Related Documentation

- **Building for Vertex AI:** [VERTEX_TRAINING.md](VERTEX_TRAINING.md)
- **Checkpoint Management:** [CHECKPOINT_RESUMPTION.md](CHECKPOINT_RESUMPTION.md)
- **Training Guide:** [TRAINING.md](TRAINING.md)
