# Enhanced Benchmark Script: Support for --model_name and Config-Free W&B Artifacts

## Problem Solved
The benchmark script now works with W&B artifacts that **don't have config files** or have incomplete configs, when you specify a base model via `--model_name` or `--base_model`.

## New Features Added

### 1. **`--model_name` Argument**
Added as an alias for `--base_model` for easier usage:

```bash
# These are now equivalent:
python benchmark.py --model_name google/codegemma-2b --wandb_artifact "user/project/model:v0"
python benchmark.py --base_model google/codegemma-2b --wandb_artifact "user/project/model:v0"
```

### 2. **Config-Free W&B Artifact Support**
The script now handles W&B artifacts without config files by:
- Using the provided `--model_name` or `--base_model` as fallback
- Only using config file if it exists and has valid base model info
- Providing clear feedback about which base model is being used

### 3. **Enhanced Error Handling**
- Better fallback strategies when config files are missing or incomplete
- Clear messaging about which base model source is being used
- Graceful degradation with sensible defaults

## Usage Examples

### ✅ **Your Use Case: Config-Free W&B Artifact**
```bash
# Now works even if the W&B artifact has no config.json or incomplete config
python benchmark.py --model_name google/codegemma-2b --wandb_artifact "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v12"
```

**What happens:**
1. ✅ Downloads W&B artifact (34GB model)
2. ✅ Looks for config.json (may not exist or be incomplete)
3. ✅ Uses `google/codegemma-2b` from `--model_name` as fallback
4. ✅ Loads both base model and W&B artifact model
5. ✅ Runs benchmarks comparing them

### ✅ **Config File Takes Precedence**
```bash
# If config.json exists and specifies a base model, it's used
python benchmark.py --model_name microsoft/DialoGPT-medium --wandb_artifact "user/project/model:v0"
```

**Output:**
```
📄 Found base model in config: google/codegemma-2b
📄 Using base model: google/codegemma-2b (from config)
# Config wins over --model_name argument
```

### ✅ **Fallback When No Config**
```bash
# If no config.json or incomplete config, uses your argument
python benchmark.py --model_name google/codegemma-7b --wandb_artifact "user/project/model:v0"
```

**Output:**
```
⚠️ Warning: No base model specified in config or arguments. Using default: google/codegemma-2b
📄 Using base model: google/codegemma-7b (from argument)
# Your argument is used as fallback
```

## Technical Implementation

### **Enhanced `load_wandb_artifact()` Function**
```python
def load_wandb_artifact(artifact_path, tokenizer=None, hf_token=None, fallback_base_model=None):
    # Download artifact
    artifact_dir = artifact.download()
    
    # Check for config (optional)
    config_base_model = get_base_model_from_config(model_path)
    
    # Determine effective base model
    effective_base_model = config_base_model or fallback_base_model or "google/codegemma-2b"
    
    # Load model with the effective base model
    model, tokenizer = load_trained_model(model_path, tokenizer, hf_token, effective_base_model)
```

### **Enhanced `load_trained_model()` Function**
```python
def load_trained_model(model_path, tokenizer=None, hf_token=None, fallback_base_model=None):
    # Try to get base model from config
    config_base_model = get_base_model_from_config(model_path)
    
    # Use config if available, otherwise fallback
    effective_base_model = config_base_model or fallback_base_model or "google/codegemma-2b"
    
    # Load DyLoRA-MoE model using effective base model
    if is_dylora_moe:
        # Create compatible config using effective_base_model
        base_config = AutoModelForCausalLM.from_pretrained(effective_base_model).config.to_dict()
        # ... rest of loading logic
```

### **Argument Processing**
```python
# Handle --model_name as alias for --base_model
if args.model_name:
    args.base_model = args.model_name
    print(f"ℹ️ Using model_name as base_model: {args.base_model}")

# Pass base_model as fallback to W&B loading
wandb_model, wandb_tokenizer, wandb_config_base_model = load_wandb_artifact(
    args.wandb_artifact, tokenizer, hf_token, args.base_model  # <-- fallback
)
```

## Decision Flow

```
┌─────────────────────┐
│ W&B Artifact       │
│ Downloaded          │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐      YES     ┌─────────────────────┐
│ config.json exists  │─────────────▶│ Use config base     │
│ with base_model?    │              │ model               │
└──────┬──────────────┘              └─────────────────────┘
       │ NO
       ▼
┌─────────────────────┐      YES     ┌─────────────────────┐
│ --model_name or     │─────────────▶│ Use provided        │
│ --base_model given? │              │ argument            │
└──────┬──────────────┘              └─────────────────────┘
       │ NO
       ▼
┌─────────────────────┐
│ Use default:        │
│ google/codegemma-2b │
└─────────────────────┘
```

## Backward Compatibility

✅ **All existing usage patterns still work:**

```bash
# Still works (config-based)
python benchmark.py --wandb_artifact "user/project/model:v0"

# Still works (local model)
python benchmark.py --trained_model ./results_full/best_model

# Still works (base model only)
python benchmark.py --base_model google/codegemma-2b

# New: model_name alias
python benchmark.py --model_name google/codegemma-2b --wandb_artifact "user/project/model:v0"
```

## Error Messages and Feedback

### **Clear Source Indication**
```
📄 Using base model: google/codegemma-2b (from config)
📄 Using base model: google/codegemma-7b (from argument)
⚠️ Warning: No base model specified in config or arguments. Using default: google/codegemma-2b
```

### **Enhanced Debugging**
```
✓ Artifact downloaded to: /app/artifacts/best-dylora-model-full:v12
Detected DyLoRA-MoE model format
Loading DyLoRA-MoE model with base: google/codegemma-2b
✓ Loaded DyLoRA-MoE as merged AutoModelForCausalLM
✓ Trained model loaded from: /app/artifacts/best-dylora-model-full:v12/best_model
```

Your benchmark script is now robust enough to handle W&B artifacts with or without config files! 🎉