# Quick Reference: GitHub Secrets Setup

## Required Secrets (3 total)

### 1. DOCKER_HUB_USERNAME
```
Value: johnrcumming001
```
**Get it:** Your Docker Hub username from https://hub.docker.com/settings/general

---

### 2. DOCKER_HUB_TOKEN
```
Create at: https://hub.docker.com/settings/security
Steps:
  1. Click "New Access Token"
  2. Description: "GitHub Actions DyLoRA-MoE"
  3. Permissions: "Read, Write, Delete"
  4. Click "Generate"
  5. Copy the token (you can't see it again!)
```

---

### 3. GCP_CREDENTIALS
```
JSON key for service account with Artifact Registry Writer role
```

**Option A - Create new service account:**
1. Go to: https://console.cloud.google.com/iam-admin/serviceaccounts?project=dylora
2. Click "Create Service Account"
   - Name: `github-actions-docker-push`
3. Grant role: `Artifact Registry Writer`
4. Create JSON key
5. Copy entire JSON contents

**Option B - Use existing service account (dylora-ece7d9f1337d.json):**
1. Grant it Artifact Registry Writer role:
   ```bash
   gcloud projects add-iam-policy-binding dylora \
     --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
     --role="roles/artifactregistry.writer"
   ```
2. Use the JSON file contents

---

## Setup Methods

### Method 1: GitHub Web UI (Easiest)
1. Go to: https://github.com/johnrcumming/DyLoRA-MoE/settings/secrets/actions
2. Click "New repository secret" for each:
   - Name: `DOCKER_HUB_USERNAME`, Value: your username
   - Name: `DOCKER_HUB_TOKEN`, Value: your token
   - Name: `GCP_CREDENTIALS`, Value: paste entire JSON

### Method 2: Automated Script
```bash
cd /Users/johncumming/work/DyLoRA
./.github/setup-secrets.sh
```

### Method 3: GitHub CLI (Manual)
```bash
# Set Docker Hub secrets
gh secret set DOCKER_HUB_USERNAME -b "johnrcumming001"
echo "YOUR_DOCKER_TOKEN" | gh secret set DOCKER_HUB_TOKEN

# Set GCP secret
gh secret set GCP_CREDENTIALS < /path/to/service-account.json
```

---

## Verification Checklist

After setup, verify:
- [ ] All 3 secrets appear at: https://github.com/johnrcumming/DyLoRA-MoE/settings/secrets/actions
- [ ] Secret names are exactly: `DOCKER_HUB_USERNAME`, `DOCKER_HUB_TOKEN`, `GCP_CREDENTIALS`
- [ ] GCP service account has "Artifact Registry Writer" role
- [ ] Docker Hub token has "Read, Write, Delete" permissions

---

## Test the Workflow

Trigger a test build:
```bash
# Commit the workflow file
git add .github/workflows/docker-build-push.yml
git commit -m "Add Docker build and push workflow"
git push

# Or trigger manually
gh workflow run docker-build-push.yml
```

Monitor at: https://github.com/johnrcumming/DyLoRA-MoE/actions

---

## Expected Results

✅ **On Success:**
- Images pushed to both registries
- Tagged with git hash and `latest`
- Available at:
  - `us-central1-docker.pkg.dev/dylora/dy-lora-training-repo/dy-lora-training-image:latest`
  - `johnrcumming001/dylora-moe:latest`

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Unable to authenticate to Docker Hub" | Regenerate token at hub.docker.com/settings/security |
| "Failed to authenticate to GCP" | Check JSON is valid & service account exists |
| "Permission denied to Artifact Registry" | Grant `roles/artifactregistry.writer` to service account |
| Workflow not triggering | Check file paths in workflow match your changes |
