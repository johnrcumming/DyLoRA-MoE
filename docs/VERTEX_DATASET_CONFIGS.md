# Dataset Configuration Options for Vertex AI Training

This document provides pre-configured dataset combinations optimized for different GPU memory constraints and training objectives.

## Current Configuration (submit_full_training.py)

**Large-Scale Training (~230k examples)**
```python
"--datasets", "code_alpaca,mbpp,evol_instruct,code_feedback"
"--train_batch_size", "1"
"--eval_batch_size", "2"
"--gradient_accumulation_steps", "128"
```

Memory: **~39GB GPU** required  
Training time: **~30-40 hours** on H100  
Effective batch size: 128

---

## Alternative Configurations

### Option 1: Medium-Scale Training (~100k examples)
**Best for: Balanced training with good diversity**

```python
"--datasets", "code_alpaca,evol_instruct,mbpp",
"--train_batch_size", "1",
"--eval_batch_size", "4",
"--gradient_accumulation_steps", "64",
```

- **Total examples**: ~88,856
- **Distribution**: 
  - evol_instruct: 70,437 (79%)
  - code_alpaca: 18,019 (20%)
  - mbpp: 400 (1%)
- **Memory**: ~25-30GB GPU
- **Training time**: ~15-20 hours on H100

### Option 2: Evol-Instruct Focus (~80k examples)
**Best for: High-quality instruction following**

```python
"--datasets", "evol_instruct,mbpp",
"--train_batch_size", "2",
"--eval_batch_size", "4",
"--gradient_accumulation_steps", "32",
```

- **Total examples**: ~70,837
- **Distribution**: 
  - evol_instruct: 70,437 (99.4%)
  - mbpp: 400 (0.6%)
- **Memory**: ~20-25GB GPU
- **Training time**: ~12-15 hours on H100
- **Note**: Use `--interleaved_sampling` for 50/50 balance

### Option 3: Default Small-Scale (~20k examples)
**Best for: Quick iteration and testing**

```python
"--datasets", "code_alpaca,mbpp",  # or omit (default)
"--train_batch_size", "4",
"--eval_batch_size", "8",
"--gradient_accumulation_steps", "16",
```

- **Total examples**: ~18,419
- **Distribution**:
  - code_alpaca: 18,019 (98%)
  - mbpp: 400 (2%)
- **Memory**: ~15-20GB GPU
- **Training time**: ~3-5 hours on H100
- **Note**: Use `--interleaved_sampling` for 50/50 balance

### Option 4: Python Specialist (~45k examples)
**Best for: Python-focused applications**

```python
"--datasets", "python_codes_25k,python_code_instructions_18k,mbpp",
"--train_batch_size", "2",
"--eval_batch_size", "4",
"--gradient_accumulation_steps", "32",
```

- **Total examples**: ~43,400
- **Distribution**:
  - python_codes_25k: 25,000 (58%)
  - python_code_instructions_18k: 18,000 (41%)
  - mbpp: 400 (1%)
- **Memory**: ~15-20GB GPU
- **Training time**: ~8-10 hours on H100

---

## How to Update submit_full_training.py

Replace the `"command"` section in `worker_pool_specs[0]["container_spec"]`:

```python
"command": [
    "python", "train.py",
    "--datasets", "YOUR_DATASET_CHOICE",  # Pick from above
    "--bf16", 
    "--num_epochs", "10",
    "--num_experts", "4",
    "--balance_coefficient", "0.01",
    "--cosine_restarts",
    "--train_batch_size", "YOUR_BATCH_SIZE",
    "--eval_batch_size", "YOUR_EVAL_BATCH",
    "--gradient_accumulation_steps", "YOUR_ACCUMULATION",
    "--early_stopping_patience", "3",
],
```

---

## GPU Memory Guidelines

| Configuration | Dataset Size | GPU Memory | Recommended Batch Settings |
|--------------|-------------|-----------|---------------------------|
| Large-Scale | ~230k | 40GB (H100) | batch=1, accum=128 |
| Medium | ~100k | 30GB (A100/H100) | batch=1, accum=64 |
| Evol-Focus | ~80k | 25GB (A100) | batch=2, accum=32 |
| Default | ~20k | 20GB (A100) | batch=4, accum=16 |
| Python Specialist | ~45k | 20GB (A100) | batch=2, accum=32 |

---

## Effective Batch Size

Formula: `effective_batch = train_batch_size × gradient_accumulation_steps × num_gpus`

- Large-scale: 1 × 128 × 1 = **128**
- Medium: 1 × 64 × 1 = **64**
- Default: 4 × 16 × 1 = **64**

Aim for effective batch size of **32-128** for stable training.

---

## Cost Optimization

### For Development/Testing
```python
"--datasets", "code_alpaca,mbpp",
"--training_subset", "10",  # Use 10% of data
"--eval_subset", "20",
"--num_epochs", "3",
```

### For Production Training
```python
"--datasets", "code_alpaca,evol_instruct,code_feedback",
"--num_epochs", "10",
"--early_stopping_patience", "3",
```

---

## Current Issue Resolution

The OOM error occurred because:
1. 4 datasets = ~230k examples (12x larger than default)
2. Multi-expert forward pass (4 experts) multiplies memory usage
3. Original batch settings (batch=1, accum=64) insufficient

**Solution Applied**: 
- Reduced eval batch: 4 → 2
- Increased accumulation: 64 → 128
- This maintains effective batch size while reducing peak memory

If OOM persists:
1. Use **Option 2 or 3** (fewer examples)
2. Or use **gradient checkpointing** (adds to train.py)
3. Or reduce `--num_experts` to 2
