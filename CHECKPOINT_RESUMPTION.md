# Checkpoint Resumption Guide

## Overview

DyLoRA-MoE now supports resuming training from Weights & Biases (W&B) artifacts, enabling you to:
- **Continue training** from where a previous run stopped (e.g., after early stopping at epoch 4)
- **Resume interrupted jobs** (e.g., spot VM preemption)
- **Transfer checkpoints** across different training environments (local → cloud, cloud → cloud)

All checkpoints are downloaded directly from W&B, eliminating the need for local file management in Docker containers.

---

## Quick Start

### 1. Find Your Checkpoint Artifact

After a training run completes, the model is automatically uploaded to W&B as an artifact named `best-dylora-model-full`.

**View your artifacts:**
```bash
# Web UI: https://wandb.ai/<username>/dylo-moe-full-training/artifacts
# Look for artifacts with type "model"
```

**Artifact naming format:**
```
<username>/<project>/<artifact-name>:<version>
```

**Example:**
```
johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0
```

### 2. Resume Training from Checkpoint

#### Option A: Cloud Training (Vertex AI)

Edit `submit_full_training.py`:
```python
# Line 33: Set the checkpoint artifact path
WANDB_CHECKPOINT_ARTIFACT = "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0"
```

Then submit:
```bash
./build_and_push.sh && python submit_full_training.py
```

#### Option B: Local Training

```bash
python train.py \
  --datasets "code_alpaca,mbpp,evol_instruct,code_feedback" \
  --wandb_checkpoint_artifact "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0" \
  --bf16 \
  --num_epochs 10 \
  --num_experts 2
```

---

## How It Works

### W&B Artifact Download

When `--wandb_checkpoint_artifact` is provided:

1. **Initialization**: `wandb.init()` authenticates using `WANDB_API_KEY` environment variable
2. **Download**: `wandb.use_artifact(artifact_path).download()` downloads checkpoint to temporary directory
3. **Resume**: HuggingFace `Trainer.train(resume_from_checkpoint=<path>)` loads checkpoint and continues training

### What Gets Restored

From the W&B artifact checkpoint:
- ✅ **Model weights**: All LoRA adapter parameters (`lora_A`, `lora_B` for each expert)
- ✅ **Router state**: Expert routing weights and maturity flags
- ✅ **Optimizer state**: Adam/AdamW momentum and variance
- ✅ **Training state**: Epoch number, global step, learning rate scheduler
- ✅ **Best metrics**: Best validation loss for early stopping continuation

### What Doesn't Get Restored (By Design)

- ❌ **Training arguments**: Specified fresh via CLI (allows changing hyperparameters)
- ❌ **Dataset selection**: Specified fresh via `--datasets` (allows dataset swapping)
- ❌ **Hardware config**: Inherits from new environment (allows cloud → local migration)

---

## Example Use Cases

### Use Case 1: Continue Training After Early Stopping

**Scenario**: Your first run stopped at epoch 4 due to early stopping (patience=3). You want to train to epoch 10.

**Solution**:
```python
# submit_full_training.py
WANDB_CHECKPOINT_ARTIFACT = "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0"

# Command already includes:
# --num_epochs 10
# --early_stopping_patience 5  # Increased patience
```

**Result**: Training resumes from epoch 4, continues to epoch 10 with extended patience.

---

### Use Case 2: Resume After Spot VM Preemption

**Scenario**: Vertex AI spot VM was preempted at epoch 2.5 during a 10-epoch run.

**Solution**:
```python
# W&B auto-saves checkpoints every N steps
# Find the latest checkpoint artifact (e.g., v3 if multiple saves occurred)
WANDB_CHECKPOINT_ARTIFACT = "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v3"
```

**Result**: Training resumes from last saved checkpoint, completing remaining 7.5 epochs.

---

### Use Case 3: Experiment with Different Datasets

**Scenario**: You trained on `code_alpaca,mbpp` (20k examples) and want to continue training on larger dataset.

**Solution**:
```python
# submit_full_training.py
WANDB_CHECKPOINT_ARTIFACT = "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0"

# Change line 55:
"--datasets", "code_alpaca,mbpp,evol_instruct,code_feedback",  # Expanded dataset
```

**Result**: Model continues from checkpoint but trains on new, larger dataset combination.

---

### Use Case 4: Transfer Cloud Checkpoint to Local Development

**Scenario**: You want to debug or experiment locally with a model trained on Vertex AI.

**Solution**:
```bash
# Download checkpoint locally
python train.py \
  --wandb_checkpoint_artifact "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0" \
  --training_subset 10 \
  --eval_subset 20 \
  --num_epochs 1 \
  --bf16
```

**Result**: Checkpoint downloads to local machine, trains on 10% subset for quick iteration.

---

## Troubleshooting

### Error: "Failed to download W&B artifact"

**Cause**: Missing or invalid `WANDB_API_KEY` environment variable.

**Solution**:
```bash
# Local
export WANDB_API_KEY=your_key_here

# Vertex AI (already configured in submit_full_training.py)
# Verify .env file contains WANDB_API_KEY
```

### Error: "Artifact not found"

**Cause**: Incorrect artifact path or artifact was deleted.

**Solution**:
```bash
# Verify artifact exists on W&B web UI
# Check artifact path format: username/project/artifact-name:version
```

### Warning: "Checkpoint from different model architecture"

**Cause**: Resuming checkpoint with different `--num_experts` than original training.

**Solution**:
```bash
# Original training: --num_experts 2
# Resume training: --num_experts 2  # MUST match
```

### Training Starts from Epoch 0 Despite Checkpoint

**Cause**: `Trainer` couldn't find optimizer state or training arguments in checkpoint.

**Solution**:
```bash
# Ensure checkpoint includes:
# - trainer_state.json
# - optimizer.pt
# - scheduler.pt
# These are automatically saved by HuggingFace Trainer
```

---

## Advanced: Checkpoint Artifact Versioning

W&B automatically versions artifacts:
- `v0`: First upload
- `v1`: Second upload (overwrites if same name)
- `v2`, `v3`, etc.: Subsequent uploads

**Find specific version:**
```python
# Latest version (recommended)
WANDB_CHECKPOINT_ARTIFACT = "johnrcumming/dylo-moe-full-training/best-dylora-model-full:latest"

# Specific version
WANDB_CHECKPOINT_ARTIFACT = "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v2"
```

---

## Best Practices

1. **Always use `:latest` for continuation**: Ensures you resume from most recent checkpoint
2. **Pin specific version for reproducibility**: Use `:v0` for exact experiment replication
3. **Test locally first**: Validate checkpoint download works before cloud submission
4. **Monitor W&B logs**: Check "Files" tab in W&B run to verify checkpoint upload succeeded
5. **Set WANDB_CHECKPOINT_ARTIFACT = None for fresh training**: Prevents accidental resumption

---

## Implementation Details

### Code Locations

- **Checkpoint download logic**: `train.py` lines 448-464
- **Resume logic**: `train.py` lines 771-777
- **CLI argument**: `train.py` line 833-836
- **Vertex AI config**: `submit_full_training.py` lines 32-34, 65

### Artifact Upload

Checkpoints are uploaded at:
1. **End of training**: `trainer.train()` completes
2. **Best model save**: When new best validation loss is achieved
3. **Manual upload**: `wandb.log_artifact()` in `train.py` lines 817-822

---

## Migration from Legacy Checkpoint System

**Old way** (local file, doesn't work in Docker):
```bash
python train.py --resume_from_checkpoint
# Looks for ./results_full/checkpoint-latest (doesn't exist in container)
```

**New way** (W&B artifact, works everywhere):
```bash
python train.py --wandb_checkpoint_artifact "username/project/artifact:version"
# Downloads from W&B automatically
```

**Compatibility**: The `--resume_from_checkpoint` flag is still supported for local development but deprecated for cloud training.

---

## Quick Reference

| Action | Command/Config |
|--------|---------------|
| **Find artifact path** | W&B UI → Project → Artifacts → Copy path |
| **Resume on Vertex AI** | Set `WANDB_CHECKPOINT_ARTIFACT` in `submit_full_training.py` |
| **Resume locally** | Add `--wandb_checkpoint_artifact "path"` to `train.py` |
| **Start fresh** | Set `WANDB_CHECKPOINT_ARTIFACT = None` |
| **Test download** | `wandb artifact get username/project/artifact:version` |

---

## FAQ

**Q: Can I resume with different hyperparameters?**  
A: Yes! Only model weights/optimizer state are restored. Learning rate, batch size, etc. use new CLI values.

**Q: Does resumption work across different GPUs (A100 → H100)?**  
A: Yes! Checkpoint is hardware-agnostic.

**Q: What happens if I change `--num_experts`?**  
A: Training will fail. Expert count must match checkpoint architecture.

**Q: Can I resume from a checkpoint trained on different datasets?**  
A: Yes, but model may need adaptation period. Consider lower learning rate for fine-tuning.

**Q: How much does checkpoint download add to job startup time?**  
A: ~1-2 minutes for typical checkpoint (~50MB). W&B caching makes subsequent downloads faster.

---

For more details, see:
- **W&B Artifacts Documentation**: https://docs.wandb.ai/guides/artifacts
- **HuggingFace Trainer Checkpointing**: https://huggingface.co/docs/transformers/main_classes/trainer#checkpoints
