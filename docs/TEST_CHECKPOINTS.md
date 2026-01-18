# Checkpoint Save/Load Testing

This directory contains test scripts to validate DyLoRA-MoE checkpoint saving and loading functionality.

## Test Scripts

### 1. Quick Test (`test_checkpoint_quick.py`)

**Fast validation** of checkpoint infrastructure without loading models.

```bash
python test_checkpoint_quick.py
```

**What it tests:**
- ✓ PEFT library availability and MoE support
- ✓ safetensors library availability
- ✓ Checkpoint directory structure creation
- ✓ File format validation

**Runtime:** ~5 seconds  
**Use when:** Verifying environment setup, quick sanity checks

---

### 2. Full Test (`test_checkpoint_save_load.py`)

**Comprehensive validation** with actual model save/load cycles.

```bash
# Run with default settings (CodeGemma-2B, 2 experts)
python test_checkpoint_save_load.py

# Custom model and expert count
python test_checkpoint_save_load.py --model google/codegemma-2b --experts 4

# Keep checkpoint after tests
python test_checkpoint_save_load.py --keep_checkpoint

# Provide HF token explicitly
python test_checkpoint_save_load.py --hf_token YOUR_TOKEN
```

**What it tests:**
1. **Checkpoint Structure** - Validates PEFT adapter directories, router files, config
2. **Save Checkpoint** - Tests `save_pretrained()` for all experts + router state
3. **Load Checkpoint** - Tests loading adapters and router into new model instance
4. **Output Consistency** - Verifies model outputs identical before/after save/load
5. **Checkpoint Size** - Ensures no merged model files (should be ~100MB, not 2.6GB)

**Runtime:** ~2-3 minutes (depends on model size)  
**Use when:** Validating code changes, before commits, release testing

---

## Test Workflow

### During Development
```bash
# 1. Quick test after environment changes
python test_checkpoint_quick.py

# 2. Full test before committing changes
python test_checkpoint_save_load.py --keep_checkpoint

# 3. Inspect checkpoint if needed
ls -lh /tmp/dylora_checkpoint_test_*/test_checkpoint/
```

### Before Release
```bash
# Test with different configurations
python test_checkpoint_save_load.py --experts 2
python test_checkpoint_save_load.py --experts 4

# Verify checkpoint size
python test_checkpoint_save_load.py | grep "Total checkpoint size"
```

---

## Expected Test Output

### Quick Test (Success)
```
============================================================
DyLoRA-MoE Quick Checkpoint Tests
============================================================

============================================================
Quick Test: PEFT Availability
============================================================
✓ PEFT library is available
✓ MoELoraConfig is available
✓ attach_router method is available

============================================================
Quick Test: Safetensors Availability
============================================================
✓ safetensors is available
✓ safetensors save/load works

============================================================
Quick Test: PEFT Adapter Structure
============================================================
✓ Created expert_0 structure
✓ Created expert_1 structure
✓ Created router state
✓ Created config.json

Validating structure:
✓ peft_adapters/expert_0/adapter_config.json
✓ peft_adapters/expert_0/adapter_model.bin
✓ peft_adapters/expert_1/adapter_config.json
✓ peft_adapters/expert_1/adapter_model.bin
✓ router.pt
✓ config.json

✓ Quick test PASSED - structure is valid

============================================================
SUMMARY
============================================================
✓ PASS: peft
✓ PASS: safetensors
✓ PASS: structure
============================================================
✓ ALL QUICK TESTS PASSED
============================================================
```

### Full Test (Success)
```
============================================================
DyLoRA-MoE Checkpoint Save/Load Tests
============================================================
Model: google/codegemma-2b
Experts: 2
============================================================

[... detailed test output ...]

============================================================
TEST SUMMARY
============================================================
✓ PASS: save
✓ PASS: structure
✓ PASS: load
✓ PASS: consistency
✓ PASS: size
============================================================
✓ ALL TESTS PASSED
============================================================

ℹ️  Test checkpoint saved at: /tmp/dylora_checkpoint_test_xyz/test_checkpoint
```

---

## Checkpoint Structure Validated

Both tests verify this structure:

```
test_checkpoint/
├── peft_adapters/
│   ├── expert_0/
│   │   ├── adapter_config.json     # PEFT adapter config
│   │   └── adapter_model.safetensors  # LoRA weights (~14MB)
│   └── expert_1/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── router.safetensors              # Router state (~18KB)
├── skill_library.safetensors       # Skill library (~1KB)
└── config.json                     # DyLoRA-MoE metadata
```

**Key validations:**
- ✓ Each expert has `adapter_config.json` + weights
- ✓ Router state saved separately (not in PEFT adapters)
- ✓ Config includes model metadata
- ✓ Total size < 500MB (no merged model)
- ✓ Using safetensors format (preferred) or PyTorch fallback

---

## Troubleshooting

### "PEFT library not available"
```bash
pip install peft
```

### "safetensors not available"
```bash
pip install safetensors
```
Tests will use PyTorch format as fallback, but safetensors is preferred.

### "Output consistency test failed"
This indicates model loading didn't restore state correctly. Check:
- Router state loading
- PEFT adapter loading order
- Device placement (CPU/CUDA/MPS)

### "Checkpoint too large"
Merged model files weren't deleted. Check:
- `PeftCheckpointCallback` is deleting `model.safetensors`
- `save_safetensors=False` in `TrainingArguments`

---

## Environment Requirements

```bash
# Required
pip install torch transformers peft

# Optional but recommended
pip install safetensors

# For testing with HuggingFace models
export HF_TOKEN=your_token_here
```

---

## Integration with CI/CD

```yaml
# Example GitHub Actions
- name: Run checkpoint tests
  run: |
    python test_checkpoint_quick.py
    python test_checkpoint_save_load.py --experts 2
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

---

## Notes

- Tests create temporary directories in `/tmp/`
- Use `--keep_checkpoint` to inspect checkpoint structure
- Quick test runs without GPU/model downloads
- Full test requires HuggingFace token for model access
- Both tests clean up after themselves (unless `--keep_checkpoint`)
