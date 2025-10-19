# DyLoRA-MoE Container Entrypoint Guide

## Overview

DyLoRA-MoE now uses a unified container entrypoint (`entrypoint.py`) that supports both **training** and **benchmarking** operations across different cloud platforms (Google Vertex AI and Vast.ai).

## Key Features

- ✅ **Unified Interface**: Single entrypoint for training and benchmarking
- ✅ **Platform Detection**: Automatically detects Vertex AI vs Vast.ai vs unknown platforms  
- ✅ **Smart Defaults**: Platform-specific batch sizes and configurations
- ✅ **Environment Validation**: Checks required environment variables
- ✅ **Real-time Streaming**: Shows training/benchmark output in real-time
- ✅ **Error Handling**: Graceful error handling with proper exit codes

---

## Basic Usage

### Container Entrypoint Syntax

```bash
python entrypoint.py {--train|--benchmark} [arguments...]
```

### Training Mode

```bash
# Basic training
python entrypoint.py --train --datasets "code_alpaca,mbpp" --num_epochs 5

# Advanced training with W&B checkpoint resumption
python entrypoint.py --train \
  --datasets "code_alpaca,mbpp,evol_instruct" \
  --wandb_checkpoint_artifact "user/project/model:v0" \
  --num_epochs 10 \
  --bf16

# Platform will auto-detect and apply appropriate batch sizes
```

### Benchmarking Mode

```bash
# Benchmark local model
python entrypoint.py --benchmark --model-path "./results/best_model" --max-samples 164

# Benchmark W&B artifact
python entrypoint.py --benchmark \
  --wandb-artifact "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0" \
  --max-samples 50

# Benchmark base model (no fine-tuning)
python entrypoint.py --benchmark --base-model --max-samples 164
```

---

## Platform-Specific Behavior

### Google Vertex AI
- **Detection**: Checks for GCP metadata server or `/var/log/gcplogs-docker-driver`
- **Optimizations**: 
  - Higher batch sizes (train_batch_size=4, eval_batch_size=4)
  - More aggressive gradient accumulation (32 steps)
  - Assumes reliable hardware and networking

### Vast.ai
- **Detection**: Checks for `VAST_INSTANCE_ID` env var or "vast" in hostname
- **Optimizations**:
  - Conservative batch sizes (train_batch_size=2, eval_batch_size=2) 
  - Higher gradient accumulation (64 steps) to maintain effective batch size
  - Assumes variable hardware quality

### Unknown Platform (Local/Other)
- **Fallback**: Uses conservative settings similar to Vast.ai
- **Manual Override**: You can override any defaults with explicit arguments

---

## Docker Usage

### Local Docker Run

#### Training
```bash
# Quick training run
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  johnrcumming/dylora-moe:latest \
  --train --datasets "code_alpaca,mbpp" --num_epochs 3 --training_subset 10

# Resume from checkpoint
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  johnrcumming/dylora-moe:latest \
  --train --wandb_checkpoint_artifact "user/project/model:v0" --num_epochs 10
```

#### Benchmarking
```bash
# Benchmark base model
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  johnrcumming/dylora-moe:latest \
  --benchmark --base-model --max-samples 20

# Benchmark W&B artifact
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  johnrcumming/dylora-moe:latest \
  --benchmark --wandb-artifact "user/project/model:v0" --max-samples 164
```

### Interactive Mode
```bash
# Get shell access to container
docker run -it --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  johnrcumming/dylora-moe:latest \
  bash

# Then run commands inside container:
python entrypoint.py --train --datasets "code_alpaca,mbpp" --num_epochs 1
python entrypoint.py --benchmark --base-model --max-samples 10
```

---

## Cloud Platform Usage

### Google Vertex AI

#### Training Job
```python
# vertex_full_training.py (updated to use entrypoint)
python vertex_full_training.py
```

#### Benchmark Job
```python
# New benchmark job submission
python vertex_benchmark_job.py --wandb-artifact "user/project/model:v0" --max-samples 164

# Or benchmark base model
python vertex_benchmark_job.py --base-model --max-samples 164
```

### Vast.ai

#### Training
```bash
# vastai_full_training.sh (updated to use entrypoint)
./vastai_full_training.sh 12345678  # where 12345678 is offer ID
```

#### Manual Benchmark on Vast.ai
```bash
# SSH into running instance and run benchmark
vastai ssh <instance_id> \
  'python entrypoint.py --benchmark --model-path ./results/best_model --max-samples 164'
```

---

## Environment Variables

### Required
- `HF_TOKEN`: HuggingFace API token for model access
- `WANDB_API_KEY`: Weights & Biases API key (required for training with logging)

### Optional
- `WANDB_API_KEY`: Optional for benchmarking-only jobs
- `VAST_INSTANCE_ID`: Auto-set by Vast.ai, helps with platform detection
- `PLATFORM`: Manual platform override (`vertex-ai`, `vast-ai`, `unknown`)

### Environment Validation
The entrypoint automatically validates environment variables and provides helpful error messages:

```
ERROR - Missing required environment variables: ['HF_TOKEN']
Please set these variables before running the container:
  -e HF_TOKEN=<your_value>
```

---

## Default Argument Behavior

The entrypoint automatically applies platform-specific defaults if arguments are not provided:

### Training Defaults

| Platform | train_batch_size | eval_batch_size | gradient_accumulation_steps |
|----------|------------------|-----------------|---------------------------|
| **Vertex AI** | 4 | 4 | 32 |
| **Vast.ai** | 2 | 2 | 64 |
| **Unknown** | 2 | 2 | 64 |

**Common training defaults** (all platforms):
- `--num_epochs 10`
- `--num_experts 2` 
- `--bf16` (if neither `--bf16` nor `--fp16` specified)

### Benchmarking Defaults

| Argument | Default Value |
|----------|---------------|
| `--max-samples` | 164 (full HumanEval) |
| `--temperature` | 0.2 |
| `--max-tokens` | 256 |

---

## Migration Guide

### Old Method (Direct Scripts)
```bash
# Training
python train.py --datasets "code_alpaca,mbpp" --bf16 --num_epochs 10

# Benchmarking  
python benchmark.py --model-path "./results/best_model" --max-samples 164
```

### New Method (Unified Entrypoint)
```bash
# Training
python entrypoint.py --train --datasets "code_alpaca,mbpp" --bf16 --num_epochs 10

# Benchmarking
python entrypoint.py --benchmark --model-path "./results/best_model" --max-samples 164
```

### Container Changes

**Old Dockerfile:**
```dockerfile
ENTRYPOINT ["python", "train.py"] 
CMD ["--datasets", "code_alpaca,mbpp", "--bf16", ...]
```

**New Dockerfile:**
```dockerfile
ENTRYPOINT ["python", "entrypoint.py"]
CMD ["--train", "--datasets", "code_alpaca,mbpp", "--bf16", ...]
```

---

## Troubleshooting

### Common Issues

#### Missing Environment Variables
```
ERROR - Missing required environment variables: ['HF_TOKEN']
```
**Solution:** Add `-e HF_TOKEN=your_token` to docker run command

#### Platform Not Detected
```
Detected platform: unknown
```
**Solution:** Platform detection failed but will use conservative defaults. For manual override:
```bash
-e PLATFORM=vertex-ai  # or vast-ai
```

#### Training Fails with OOM
```
torch.OutOfMemoryError: CUDA out of memory
```
**Solution:** Override batch size manually:
```bash
python entrypoint.py --train --train_batch_size 1 --eval_batch_size 1 --gradient_accumulation_steps 128
```

#### Benchmark Model Not Found
```
ERROR - Model path './results/best_model' does not exist
```
**Solution:** Check model path or use W&B artifact instead:
```bash
python entrypoint.py --benchmark --wandb-artifact "user/project/model:v0"
```

### Debug Mode

For debugging, you can override the entrypoint:
```bash
# Get shell access
docker run -it --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  --entrypoint bash \
  johnrcumming/dylora-moe:latest

# Inside container, run manually:
python entrypoint.py --train --datasets "code_alpaca,mbpp" --num_epochs 1 --training_subset 10
```

---

## Architecture Benefits

### Before (Separate Scripts)
- ❌ Different commands for different platforms
- ❌ Manual batch size tuning required
- ❌ No platform-specific optimizations
- ❌ Separate Docker configurations needed

### After (Unified Entrypoint)
- ✅ Single command interface across platforms
- ✅ Automatic platform detection and optimization
- ✅ Consistent environment validation
- ✅ Unified container for training and benchmarking
- ✅ Easier cloud deployment automation

---

## Advanced Usage

### Custom Platform Detection Override
```bash
# Force Vertex AI optimizations on any platform
python entrypoint.py --train -e PLATFORM=vertex-ai --datasets "code_alpaca,mbpp"

# Force conservative settings
python entrypoint.py --train -e PLATFORM=vast-ai --datasets "code_alpaca,mbpp"
```

### Benchmark Multiple Models
```bash
# Benchmark base model
python entrypoint.py --benchmark --base-model --max-samples 164

# Benchmark fine-tuned model
python entrypoint.py --benchmark --wandb-artifact "user/project/model:v0" --max-samples 164

# Compare results manually or with custom scripts
```

### Integration with CI/CD
```yaml
# GitHub Actions example
- name: Run DyLoRA Training
  run: |
    docker run --rm --gpus all \
      -e HF_TOKEN=${{ secrets.HF_TOKEN }} \
      -e WANDB_API_KEY=${{ secrets.WANDB_API_KEY }} \
      johnrcumming/dylora-moe:latest \
      --train --datasets "code_alpaca,mbpp" --num_epochs 3 --training_subset 10

- name: Run Benchmark
  run: |
    docker run --rm --gpus all \
      -e HF_TOKEN=${{ secrets.HF_TOKEN }} \
      johnrcumming/dylora-moe:latest \
      --benchmark --base-model --max-samples 20
```

---

## Related Files

- **[entrypoint.py](entrypoint.py)**: Main unified entrypoint script
- **[vertex_full_training.py](vertex_full_training.py)**: Vertex AI training job submission (updated)
- **[vertex_benchmark_job.py](vertex_benchmark_job.py)**: Vertex AI benchmark job submission (new)
- **[vastai_full_training.sh](vastai_full_training.sh)**: Vast.ai training setup (updated)
- **[Dockerfile](Dockerfile)**: Container configuration (updated)
- **[train.py](train.py)**: Core training script (unchanged API)
- **[benchmark.py](benchmark.py)**: Core benchmarking script (unchanged API)

The entrypoint system maintains full backward compatibility while providing a unified, platform-aware interface for containerized deployments.