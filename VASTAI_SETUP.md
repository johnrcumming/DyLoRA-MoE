# Vast.ai Training Setup Guide

## Overview

This guide explains how to run DyLoRA-MoE training on Vast.ai, a GPU marketplace offering affordable cloud GPUs.

---

## Prerequisites

### 1. Vast.ai Account

1. Sign up at https://vast.ai
2. Add credits to your account
3. Install Vast.ai CLI:
   ```bash
   pip install vastai
   ```

### 2. API Key Setup

Configure Vast.ai CLI with your API key:
```bash
vastai set api-key YOUR_VASTAI_API_KEY
```

### 3. Environment Variables

Ensure your `.env` file contains:
```bash
WANDB_API_KEY=your_wandb_key
HF_TOKEN=your_huggingface_token
DOCKER_TOKEN=your_docker_token  # Only needed for private repos
```

---

## Quick Start

### 1. Find a GPU Instance

Search for available H100 GPUs:
```bash
# Search for H100 80GB GPUs with at least 200GB disk
vastai search offers 'gpu_name=H100 num_gpus=1 disk_space>=200'

# Look for offers with good reliability score and bandwidth
# Note the OFFER_ID from the results (e.g., 12345678)
```

**Recommended specs:**
- GPU: H100 80GB, A100 80GB, or RTX 4090
- RAM: 64GB+
- Disk: 200GB+
- Bandwidth: 500 Mbps+
- Reliability: 0.95+

### 2. Launch Training

```bash
# Make script executable (first time only)
chmod +x vastai_full_training.sh

# Launch training with your chosen offer ID
./vastai_full_training.sh 12345678
```

The script will:
1. Load environment variables from `.env`
2. Validate required keys (WANDB_API_KEY, HF_TOKEN)
3. Create Vast.ai instance with your offer ID
4. Pull Docker image from Docker Hub
5. Start training automatically

---

## Usage

### Basic Usage

```bash
./vastai_full_training.sh <OFFER_ID>
```

**Example:**
```bash
./vastai_full_training.sh 12345678
```

### Monitoring Training

```bash
# List your instances
vastai show instances

# Get instance ID from the output, then SSH into instance
vastai ssh-url <INSTANCE_ID>
ssh <connection_string_from_above>

# View training logs
docker logs -f <container_id>

# Or if training is running directly:
journalctl -f
```

### Managing Instances

```bash
# List all your instances
vastai show instances

# Stop an instance
vastai stop instance <INSTANCE_ID>

# Start a stopped instance
vastai start instance <INSTANCE_ID>

# Destroy an instance
vastai destroy instance <INSTANCE_ID>
```

---

## Configuration

### Docker Image

The script uses the public Docker Hub image:
```
johnrcumming001/dylora-moe:latest
```

To use a different version:
```bash
# Edit vastai_full_training.sh, line 43
--image johnrcumming001/dylora-moe:v1.0.0 \
```

### Training Parameters

The training parameters are configured in the Dockerfile's CMD. To customize:

1. **Create a custom startup script:**
   ```bash
   # In the instance, create run_training.sh
   #!/bin/bash
   python train.py \
     --datasets "code_alpaca,mbpp" \
     --num_experts 4 \
     --bf16 \
     --num_epochs 5
   ```

2. **Or override CMD in vastai_full_training.sh:**
   ```bash
   vastai create instance "$OFFER_ID" \
     --image johnrcumming001/dylora-moe:latest \
     --env "-e WANDB_API_KEY=$WANDB_API_KEY -e HF_TOKEN=$HF_TOKEN" \
     --disk 200 \
     --onstart-cmd "python train.py --datasets code_alpaca,mbpp --bf16 --num_epochs 5"
   ```

### Disk Space

Default: 200GB (set via `--disk 200`)

Adjust based on your needs:
- **Small datasets** (code_alpaca + mbpp): 50GB
- **Medium datasets** (+ evol_instruct): 100GB
- **Large datasets** (all 4 datasets): 200GB
- **With checkpoints saved**: 300GB+

---

## Cost Estimation

Typical costs on Vast.ai (as of October 2025):

| GPU | RAM | $/hour | 15h training | Notes |
|-----|-----|--------|--------------|-------|
| **H100 80GB** | 128GB | $1.50-2.50 | $22.50-37.50 | Best performance |
| **A100 80GB** | 128GB | $1.00-1.50 | $15.00-22.50 | Good balance |
| **RTX 4090** | 64GB | $0.40-0.80 | $6.00-12.00 | Budget option |
| **RTX 3090** | 48GB | $0.25-0.50 | $3.75-7.50 | May OOM on large datasets |

**Cost-saving tips:**
1. Use "interruptible" instances for 50% discount
2. Choose locations with lower bandwidth costs
3. Stop instances when not training
4. Use smaller datasets for testing

---

## Troubleshooting

### Error: "WANDB_API_KEY not found in .env"

**Solution:**
```bash
# Check .env file exists
cat .env

# Ensure it contains:
WANDB_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

### Error: "command not found: vastai"

**Solution:**
```bash
# Install Vast.ai CLI
pip install vastai

# Verify installation
vastai --version
```

### Error: "API key not set"

**Solution:**
```bash
# Set Vast.ai API key
vastai set api-key YOUR_VASTAI_API_KEY

# Find your API key at: https://vast.ai/console/account/
```

### Instance Creation Fails

**Causes:**
1. Offer already taken (try another offer ID)
2. Insufficient credits (add funds to account)
3. Offer no longer available (search again)

**Solution:**
```bash
# Search for new offers
vastai search offers 'gpu_name=H100 num_gpus=1 disk_space>=200'

# Try a different offer ID
./vastai_full_training.sh <NEW_OFFER_ID>
```

### Docker Image Pull Fails

**Error:** `unauthorized: authentication required`

**Solution (for public image):**
```bash
# Remove --login flag from vastai_full_training.sh
# Public images don't need authentication
```

**Solution (for private image):**
```bash
# Ensure DOCKER_TOKEN is in .env
echo "DOCKER_TOKEN=dckr_pat_..." >> .env

# Rerun script
./vastai_full_training.sh <OFFER_ID>
```

### Training Stops Unexpectedly

**Possible causes:**
1. Instance preempted (interruptible)
2. Out of credits
3. Network disconnection

**Solution:**
```bash
# Check instance status
vastai show instances

# If stopped, restart
vastai start instance <INSTANCE_ID>

# Resume training from checkpoint
# (requires --wandb_checkpoint_artifact to be set in Dockerfile)
```

### OOM Error During Training

**Solution:**
```bash
# Use smaller batch size
# Edit Dockerfile or override:
vastai create instance "$OFFER_ID" \
  --image johnrcumming001/dylora-moe:latest \
  --env "-e WANDB_API_KEY=$WANDB_API_KEY -e HF_TOKEN=$HF_TOKEN" \
  --disk 200 \
  --onstart-cmd "python train.py --train_batch_size 2 --eval_batch_size 1 --bf16"
```

---

## Advanced Usage

### Custom Training Script

```bash
# Create custom training command
CUSTOM_CMD="python train.py --datasets evol_instruct --num_experts 4 --bf16"

# Launch with custom command
vastai create instance <OFFER_ID> \
  --image johnrcumming001/dylora-moe:latest \
  --env "-e WANDB_API_KEY=$WANDB_API_KEY -e HF_TOKEN=$HF_TOKEN" \
  --disk 200 \
  --onstart-cmd "$CUSTOM_CMD"
```

### Resume from Checkpoint

To resume from a W&B checkpoint:

1. **Edit Dockerfile** to include checkpoint artifact:
   ```dockerfile
   CMD [ \
       "python", "train.py", \
       "--wandb_checkpoint_artifact", "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0", \
       "--datasets", "code_alpaca,mbpp", \
       "--bf16", \
       "--num_epochs", "10" \
   ]
   ```

2. **Rebuild and push image:**
   ```bash
   ./build_and_push.sh
   ```

3. **Launch instance:**
   ```bash
   ./vastai_full_training.sh <OFFER_ID>
   ```

### Multiple Instances

Run multiple experiments in parallel:

```bash
# Launch instance 1: Small dataset
./vastai_full_training.sh <OFFER_ID_1>

# Launch instance 2: Large dataset
# (modify Dockerfile first with different --datasets)
./vastai_full_training.sh <OFFER_ID_2>

# Monitor both
vastai show instances
```

---

## Comparison: Vast.ai vs Vertex AI

| Feature | Vast.ai | Vertex AI |
|---------|---------|-----------|
| **Cost** | $0.40-2.50/h | $3-5/h |
| **Setup** | CLI (simple) | GCP account + Docker registry |
| **Reliability** | Variable (depends on host) | High (99.9% SLA) |
| **GPU Options** | Wide variety | Limited to GCP offerings |
| **Spot Pricing** | Always available | Spot VMs 50% off |
| **Best For** | Quick experiments, budget | Production, enterprise |

---

## Security Best Practices

1. ✅ **Use .env for secrets** (not hardcoded in scripts)
2. ✅ **Never commit .env to git** (already in .gitignore)
3. ✅ **Rotate API keys regularly**
4. ✅ **Use Docker Hub public images** (no auth needed)
5. ✅ **Monitor instance costs** (set budget alerts on Vast.ai)

---

## Quick Reference

### Essential Commands

```bash
# Search for GPUs
vastai search offers 'gpu_name=H100 num_gpus=1'

# Launch training
./vastai_full_training.sh <OFFER_ID>

# List instances
vastai show instances

# SSH into instance
vastai ssh-url <INSTANCE_ID>

# Stop instance
vastai stop instance <INSTANCE_ID>

# Destroy instance
vastai destroy instance <INSTANCE_ID>
```

### File Locations

- **Script:** `vastai_full_training.sh`
- **Environment:** `.env`
- **Docker Image:** `johnrcumming001/dylora-moe:latest`
- **Documentation:** This file (`VASTAI_SETUP.md`)

---

## Related Documentation

- **Docker Setup:** [DOCKER_SETUP.md](DOCKER_SETUP.md)
- **Checkpoint Resumption:** [CHECKPOINT_RESUMPTION.md](CHECKPOINT_RESUMPTION.md)
- **Training Guide:** [TRAINING.md](TRAINING.md)
- **Vertex AI Setup:** [VERTEX_TRAINING.md](VERTEX_TRAINING.md)
