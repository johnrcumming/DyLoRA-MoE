# DyLoRA-MoE Dataset Selection Feature

## Summary

Added comprehensive dataset selection feature to DyLoRA-MoE training pipeline with 7 new high-quality code generation datasets.

## What's New

### 1. New Datasets Added (`data/prepare_data.py`)

- **evol_instruct** (80k examples) - High-quality evolved code instructions ⭐
- **code_feedback** - Filtered code instructions with multi-language support ⭐
- **python_codes_25k** - Python-focused code examples
- **python_code_instructions_18k** - Python instructions in Alpaca format
- **python_code_23k_sharegpt** - Conversational Python code (ChatGPT style)

### 2. Dataset Registry System

New centralized registry in `prepare_data.py`:
```python
AVAILABLE_DATASETS = {
    'code_alpaca': download_code_alpaca,
    'mbpp': download_mbpp,
    'evol_instruct': download_evol_instruct,
    # ... more datasets
}

# Easy access via get_dataset() function
dataset = get_dataset('evol_instruct', with_validation=True)
```

### 3. Command-Line Dataset Selection

New `--datasets` argument in `train.py`:

```bash
# Default (backwards compatible)
python train.py --bf16 --num_epochs 10
# Uses: code_alpaca,mbpp

# Custom selection
python train.py --datasets "evol_instruct,mbpp" --bf16 --num_epochs 10

# Multiple datasets
python train.py --datasets "code_alpaca,evol_instruct,code_feedback" --bf16
```

### 4. Automatic Format Detection

The training pipeline automatically handles different dataset schemas:
- Alpaca format (instruction/input/output)
- Code Feedback format (query/answer)
- ShareGPT format (conversations)
- MBPP format (text)

### 5. Dataset Sampling Strategies

**Interleaved (Balanced)** - Works with exactly 2 datasets:
```bash
python train.py --datasets "code_alpaca,mbpp" --interleaved_sampling
# Result: 50% code_alpaca, 50% mbpp
```

**Concatenation (Size-based)** - Default for 1 or 3+ datasets:
```bash
python train.py --datasets "evol_instruct,code_feedback,mbpp"
# Result: Proportional to dataset sizes (~80k, ~Xk, ~1k)
```

## Files Modified

1. **`data/prepare_data.py`**
   - Added 5 new dataset download functions
   - Created `AVAILABLE_DATASETS` registry
   - Added `get_dataset()` helper function

2. **`train.py`**
   - Added `--datasets` argument
   - Modified dataset loading logic to support multiple datasets
   - Added automatic format detection in `extract_text_from_dataset()`
   - Updated imports to include registry

3. **`README.md`**
   - Added dataset examples to custom configuration section
   - Listed available datasets
   - Added new command-line arguments

4. **`DATASETS.md`** (NEW)
   - Comprehensive dataset documentation
   - Usage examples
   - Recommended configurations
   - Instructions for adding new datasets

## Usage Examples

### Quick Start (Default)
```bash
python train.py --bf16 --num_epochs 10
```

### Large-Scale Training (~180k examples)
```bash
python train.py \
  --datasets "code_alpaca,evol_instruct,code_feedback,mbpp" \
  --bf16 --num_epochs 10 \
  --balance_coefficient 0.01 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 32
```

### Python Specialist (~60k examples)
```bash
python train.py \
  --datasets "python_codes_25k,python_code_instructions_18k,mbpp" \
  --bf16 --num_epochs 10
```

### Vertex AI Cloud Training
Update `submit_full_training.py`:
```python
"command": [
    "python", "train.py",
    "--datasets", "evol_instruct,mbpp,code_feedback",
    "--bf16",
    "--num_epochs", "10",
    # ... other args
],
```

## Backward Compatibility

✅ Fully backward compatible - defaults to `"code_alpaca,mbpp"` if `--datasets` is not specified.

Existing commands continue to work:
```bash
python train.py --bf16 --num_epochs 10
# Equivalent to:
python train.py --datasets "code_alpaca,mbpp" --bf16 --num_epochs 10
```

## Testing

Tested components:
- ✅ Dataset registry system
- ✅ Argument parsing
- ✅ Dataset validation
- ✅ Multiple dataset loading
- ✅ Format detection logic

## Next Steps

To use in production:

1. **Test locally** with subset:
   ```bash
   python train.py --datasets "evol_instruct,mbpp" \
     --training_subset 10 --eval_subset 20 --num_epochs 3 --bf16
   ```

2. **Update cloud config** in `submit_full_training.py` to use preferred datasets

3. **Monitor training** - larger datasets may require:
   - Reduced batch size
   - Increased gradient accumulation steps
   - More epochs for convergence

## Documentation

See [DATASETS.md](DATASETS.md) for:
- Complete dataset descriptions
- Recommended configurations
- Adding new datasets
- Format handling details
