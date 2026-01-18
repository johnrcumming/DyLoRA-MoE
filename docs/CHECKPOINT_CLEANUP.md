# Checkpoint Cleanup - PEFT-Only Checkpoints

## Problem

Previously, `trainer.save_model()` was creating **massive merged model files** in every checkpoint directory:
- `model.safetensors` or `pytorch_model.bin` (~2.5GB each)
- Full merged weights that duplicate the base model
- **Polluted checkpoints** with redundant data when PEFT adapters were already saved separately

This wasted **gigabytes of disk space** per checkpoint with no benefit.

## Solution

Modified `PeftCheckpointCallback` to **prevent full model saving** by setting `control.should_save = False`:

```python
def on_save(self, args, state, control, **kwargs):
    # Save PEFT adapters only
    save_lora_experts(model, peft_adapters_dir)
    save_dylo_moe_state(model, dylo_moe_state_dir)
    
    # CRITICAL: Prevent Trainer from saving full merged model
    control.should_save = False
    
    return control
```

## Checkpoint Structure (After Fix)

Each checkpoint now contains **ONLY**:

```
checkpoint-1000/
├── peft_adapters/          # ~64MB (4 experts × 16MB each)
│   ├── expert_0/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── expert_1/
│   ├── expert_2/
│   └── expert_3/
├── dylo_moe_state/         # ~1MB
│   ├── config.json
│   ├── router.pt
│   └── skill_library.pt
├── config.json             # Minimal metadata
├── trainer_state.json      # Training state
├── optimizer.pt            # Optimizer state
└── scheduler.pt            # LR scheduler state
```

**Total size**: ~100MB per checkpoint (vs ~2.6GB before!)

**No more**:
- ❌ `model.safetensors` (2.5GB merged model)
- ❌ `pytorch_model.bin` (alternative format)
- ❌ Redundant full model weights

## Loading PEFT Checkpoints

`benchmark.py` already supports loading these PEFT-only checkpoints:

```bash
# Load PEFT checkpoint for benchmarking
python benchmark.py --trained_model ./results_full/checkpoint-1000

# benchmark.py detects peft_adapters/ and dylo_moe_state/ directories
# Reconstructs DyLoRA-MoE from PEFT adapters automatically
```

Loading logic in `benchmark.py`:
1. Detects `peft_adapters/` and `dylo_moe_state/` subdirectories
2. Reads config from `dylo_moe_state/config.json`
3. Reconstructs `DyLoRA_MoE` with correct hyperparameters
4. Loads each expert's adapter weights from `peft_adapters/expert_{id}/`
5. Loads router state from `dylo_moe_state/router.pt`
6. Returns fully functional MoE model with routing

## Benefits

✅ **26x smaller checkpoints** (~100MB vs ~2.6GB)  
✅ **No redundant data** - only PEFT adapters saved  
✅ **Faster checkpoint saving** - less I/O  
✅ **Cleaner directory structure** - clear what's needed  
✅ **Full MoE functionality preserved** - routing works perfectly  

## Migration

**Old checkpoints** (with merged models):
- Still loadable via `benchmark.py` (falls back to merged model loading)
- Can be manually cleaned: delete `model.safetensors`/`pytorch_model.bin` if PEFT dirs exist

**New checkpoints** (PEFT-only):
- Automatically created by updated `PeftCheckpointCallback`
- Require `benchmark.py` with PEFT loading support (already implemented)

## Training Output

New checkpoint save messages:
```
💾 Saving PEFT-only checkpoint-1000...
   ✓ PEFT adapters saved to checkpoint-1000/peft_adapters
   ✓ DyLoRA-MoE state saved to checkpoint-1000/dylo_moe_state
   ✓ Created minimal config.json (PEFT-only checkpoint)
✓ PEFT-only checkpoint complete for step 1000
   (Full model weights NOT saved to reduce disk usage)
```

Training initialization message:
```
✓ PEFT-only checkpoint saving enabled
  • Checkpoints will contain PEFT adapters + router state ONLY
  • Full merged model weights will NOT be saved (saves disk space)
  • Use benchmark.py to load PEFT checkpoints for evaluation
```

## Technical Details

### How `control.should_save` Works

The `TrainerControl` object's `should_save` flag tells the Trainer whether to save the model:
- `control.should_save = True` (default): Trainer calls `model.save_pretrained()`
- `control.should_save = False`: Trainer skips model saving, only saves trainer state

By setting `should_save = False` in `on_save()`, we prevent the wasteful merged model saving while still allowing the Trainer to save optimizer/scheduler state.

### Why This Is Safe

1. **PEFT adapters are sufficient** - they contain all trainable parameters
2. **Base model is frozen** - no need to save unchanged weights
3. **Router state is saved separately** - preserved in `dylo_moe_state/`
4. **Training can resume** - optimizer/scheduler state still saved
5. **Benchmarking works** - `benchmark.py` reconstructs full model from PEFT

### Resume Training from PEFT Checkpoint

Training resumption still works with `--resume_from_checkpoint`:
```bash
python train.py --resume_from_checkpoint ./results_full/checkpoint-1000
```

The Trainer loads optimizer/scheduler state, and the model is reconstructed from PEFT adapters automatically.
