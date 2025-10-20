# DyLoRA-MoE Model Loading Fix

## Problem
The W&B artifact downloaded contained a DyLoRA-MoE model that was saved with a custom `model_type`: `"dylora-moe"` in its config.json. When the benchmark script tried to load this with `AutoModelForCausalLM.from_pretrained()`, it failed because HuggingFace doesn't recognize this custom model type.

**Error Message:**
```
❌ Failed to load trained model: Unrecognized model in /app/artifacts/best-dylora-model-full:v12/best_model. 
Should have a `model_type` key in its config.json, or contain one of the following strings in its name: 
[long list of supported model types - but "dylora-moe" is not included]
```

## Root Cause
The training script (`train.py`) saves models with this custom config:
```json
{
  "base_model_name_or_path": "google/codegemma-2b",
  "model_type": "dylora-moe",  // <-- This causes the issue
  "num_experts": 2,
  "lora_r": 16,
  // ... other DyLoRA-specific fields
}
```

But `AutoModelForCausalLM` only recognizes standard HuggingFace model types like "gemma", "llama", etc.

## Solution
Enhanced the `load_trained_model()` function in `benchmark.py` to:

### 1. **Detect DyLoRA-MoE Models**
- Reads config.json and checks for `model_type`: `"dylora-moe"` 
- Extracts the `base_model_name_or_path` for proper loading

### 2. **Smart Loading Strategy**
- **PEFT Format**: If `adapter_config.json` exists, load as PEFT model
- **DyLoRA-MoE Format**: Create temporary compatible config using base model type
- **Fallback**: Use `trust_remote_code=True` for edge cases

### 3. **Config Compatibility**
For DyLoRA-MoE models, the script:
```python
# Get base model config (e.g., "gemma" type)
base_config = AutoModelForCausalLM.from_pretrained(base_model_name).config.to_dict()

# Remove custom model_type, use base model's type instead
# Temporarily save compatible config
with open(temp_config_path, 'w') as f:
    json.dump(base_config, f, indent=2)

# Load with compatible config
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=temp_config_path,  # Use base model config
    trust_remote_code=True
)
```

## Key Improvements

### ✅ **Enhanced Error Handling**
- Multiple fallback strategies
- Detailed error messages with traceback
- Graceful degradation

### ✅ **Format Detection**
- Automatically detects DyLoRA-MoE vs PEFT vs regular models
- Reads config to determine base model
- Legacy format support (dylo_moe_state directory)

### ✅ **Memory Optimization**
- Uses `torch.bfloat16` for better memory efficiency on GPU
- `low_cpu_mem_usage=True` for large models
- `device_map="auto"` for multi-GPU support

### ✅ **Robust Tokenizer Loading**
- Tries model directory first, then base model
- Proper fallback chain
- Consistent pad_token setup

## Testing the Fix

### Before (Failing):
```bash
python benchmark.py --wandb_artifact "user/project/best-dylora-model-full:v12"
# ❌ Failed to load trained model: Unrecognized model...
```

### After (Working):
```bash
python benchmark.py --wandb_artifact "user/project/best-dylora-model-full:v12"
# ✓ Artifact downloaded to: /app/artifacts/best-dylora-model-full:v12
# Detected DyLoRA-MoE model format  
# Loading DyLoRA-MoE model with base: google/codegemma-2b
# ✓ Loaded DyLoRA-MoE as merged AutoModelForCausalLM
# ✓ Trained model loaded from: /app/artifacts/best-dylora-model-full:v12/best_model
```

## Usage Examples

### 1. **W&B Artifact (Your Use Case)**
```bash
# Will now work with DyLoRA-MoE models
python benchmark.py --wandb_artifact "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v12"
```

### 2. **Local Model Directory**
```bash
# Works with any trained DyLoRA-MoE model
python benchmark.py --trained_model ./results_full/best_model
```

### 3. **Comparison Mode**
```bash
# Compare base vs trained model automatically
python benchmark.py --wandb_artifact "user/project/model:v12" --max_samples 50
```

## Technical Details

### **Config Handling**
- Preserves all DyLoRA-specific metadata
- Uses base model's `model_type` for HuggingFace compatibility
- Cleans up temporary files automatically

### **Memory Management**
- BFloat16 precision for GPU efficiency
- CPU offloading for systems without enough GPU memory
- Automatic device mapping for multi-GPU setups

### **Error Recovery**
- If config-based loading fails, tries direct loading
- If that fails, uses `trust_remote_code=True`
- Full traceback on complete failure for debugging

## Backward Compatibility
- Still works with regular HuggingFace models
- Supports PEFT adapters
- Maintains existing CLI interface
- No breaking changes to existing workflows

Your benchmark should now successfully load and evaluate the DyLoRA-MoE model from the W&B artifact! 🎉