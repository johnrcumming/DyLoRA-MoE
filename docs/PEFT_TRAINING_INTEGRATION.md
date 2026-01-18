# PEFT Training Integration with Complete MoE Support

## Overview

DyLoRA-MoE uses **PEFT library's training infrastructure** and extends it to support complete MoE-PEFT checkpoints including router state. This provides:

- ✅ **Standard PEFT adapter format** - Uses `save_pretrained()` for LoRA weights
- ✅ **Extended MoE support** - Saves router alongside adapters (PEFT pattern)
- ✅ **No custom trainer needed** - Works with standard Hugging Face `Trainer`
- ✅ **Complete checkpoints** - All MoE components saved in one location
- ✅ **Memory efficient** - All experts share frozen base model weights
- ✅ **Production-ready** - Leverages battle-tested PEFT + custom MoE extensions

## PEFT's MoE Support Status

### What PEFT Provides
- ✅ `MoELoraConfig` - Configuration for MoE training
- ✅ `attach_router()` - Method to attach router module to model
- ✅ `get_routing_weights()` - Convenience method for router inference
- ✅ Multiple adapter management - `add_adapter()`, `set_adapter()`

### What PEFT Does NOT Provide (Yet)
- ❌ Router state in `save_pretrained()` - Only saves LoRA adapters
- ❌ Router loading in `from_pretrained()` - Requires manual loading
- ❌ Complete MoE checkpoint management

### Our Extension
We extend PEFT's checkpoint saving to include router state using PEFT's conventions:
- Adapters saved via `save_pretrained()` (standard PEFT)
- Router saved as `router.safetensors` (following PEFT naming patterns)
- Skill library saved as `skill_library.safetensors` (DyLoRA-specific)

## Checkpoint Structure

### Complete MoE-PEFT Format
```
checkpoint-{step}/
├── peft_adapters/
│   ├── expert_0/
│   │   ├── adapter_config.json     # PEFT adapter configuration
│   │   └── adapter_model.safetensors   # Expert 0 LoRA weights (~14MB)
│   └── expert_1/
│       ├── adapter_config.json
│       └── adapter_model.safetensors   # Expert 1 LoRA weights (~14MB)
├── router.safetensors              # Router state (NEW - extends PEFT)
├── skill_library.safetensors       # Skill library state (DyLoRA-specific)
├── config.json                     # DyLoRA-MoE metadata
├── optimizer.pt                    # Training state
├── scheduler.pt
└── trainer_state.json
```

### What Gets Saved

**PEFT Native (`save_pretrained()`):**
- ✅ LoRA adapter weights (lora_A, lora_B matrices)
- ✅ Adapter configuration (adapter_config.json)
- ✅ Bias terms (if configured)
- ✅ DoRA magnitude vectors (if using DoRA)

**Our MoE Extension:**
- ✅ Router state dict (router.safetensors)
- ✅ Skill library state dict (skill_library.safetensors)
- ✅ MoE metadata (config.json)

**What's NOT Saved (Disk Space Savings):**
- ❌ `model.safetensors` (11GB) - **deleted automatically**
- ❌ `pytorch_model.bin` (11GB) - **deleted automatically**
- ✅ Total savings: **~11GB per checkpoint** (26x reduction!)

## Loading Checkpoints

### With PEFT Library
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("google/codegemma-2b")

# Load expert adapters using PEFT
for expert_id in range(num_experts):
    adapter_path = f"checkpoint-836/peft_adapters/expert_{expert_id}"
    if expert_id == 0:
        model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=f"expert_{expert_id}")
    else:
        model.load_adapter(adapter_path, adapter_name=f"expert_{expert_id}")
```

### With DyLoRA_MoE (Existing)
```python
# Existing loading logic still works
model = DyLoRA_MoE.from_pretrained("checkpoint-836")
```

## Training Workflow

### 1. Initialization
- `DyLoRA_MoE` creates `ExpertManager`
- `ExpertManager.create_expert()` wraps base model with `get_peft_model()`
- Additional experts added with `add_adapter()` (share base weights)

### 2. Training
- Standard `Trainer` handles forward/backward passes
- PEFT ensures only adapter parameters receive gradients
- Router learns from auxiliary load balancing loss

### 3. Checkpointing
- `PeftCheckpointCallback.on_save()` triggered at epoch end
- Each expert adapter saved with `save_pretrained()`
- Router/skill library saved separately
- Merged model files deleted to save disk space

### 4. Resumption
- Load checkpoint with `--resume_from_checkpoint checkpoint-{step}`
- PEFT adapters loaded automatically
- Training state (optimizer, scheduler) restored

## PEFT Library Features Used

1. **`get_peft_model()`** - Wraps base model with LoRA adapters
2. **`add_adapter()`** - Adds new expert adapters (shares base weights)
3. **`set_adapter()`** - Switches active expert for single-expert inference
4. **`save_pretrained()`** - Saves adapter weights and configuration
5. **`PeftMixedModel`** - Manages multiple adapters simultaneously

## Benefits

### Memory Efficiency
- **Base model**: 2.5B params, frozen (shared across all experts)
- **Per expert**: ~16M params (LoRA adapters only)
- **4 experts total**: ~2.56B params (vs 10B for 4 separate models)

### Storage Efficiency
- **Old checkpoints**: ~2.6GB (merged model + adapters)
- **New checkpoints**: ~100MB (adapters only)
- **26x reduction** in checkpoint size

### Training Stability
- PEFT handles gradient flow automatically
- No manual parameter freezing required
- Compatible with mixed precision training (bf16/fp16)

### Ecosystem Compatibility
- Works with Hugging Face `Trainer`
- Compatible with PEFT utilities (`merge_and_unload()`, etc.)
- Standard checkpoint format for sharing/deployment

## Migration Notes

### Existing Checkpoints
Old checkpoints saved with custom `save_lora_experts()` are still supported via the legacy loading path in `ExpertManager.load_expert_weights()`.

### New Checkpoints
All new checkpoints use PEFT's native format with `adapter_config.json` + `adapter_model.bin`.

### Backward Compatibility
The custom `save_lora_experts()` and `load_lora_experts()` functions are retained in `dylo_moe/utils.py` for backward compatibility but are no longer used for new checkpoints.

## Future Enhancements

1. **MoELoraConfig**: PEFT library now supports `MoELoraConfig` for native MoE training
2. **LoRA-FA Optimizer**: PEFT provides specialized optimizer for LoRA training
3. **Quantized Training**: Can combine with `prepare_model_for_kbit_training()` for 8-bit/4-bit training

## References

- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [PEFT MoE Support](https://github.com/huggingface/peft#mixture-of-experts-moe-support)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
