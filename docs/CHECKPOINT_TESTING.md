# DyLoRA-MoE Checkpoint Testing Guide

Comprehensive testing infrastructure for validating PEFT-based checkpoint save/load functionality.

## Test Scripts Overview

| Script | Purpose | Runtime | Memory | Use Case |
|--------|---------|---------|--------|----------|
| `test_checkpoint_quick.py` | Library checks + structure validation | ~5s | <1GB | Quick environment validation |
| `test_checkpoint_minimal.py` | **Save + structure (no reload)** | ~30s | ~5GB | **Recommended for development** |
| `test_checkpoint_save_load.py` | Full end-to-end with consistency check | ~3-5min | ~12GB | Pre-release validation (may OOM) |

## 1. Quick Test (Environment Validation)

Ultra-fast test without loading any models.

```bash
python test_checkpoint_quick.py
```

**Tests:**
- ✓ PEFT library with MoELoraConfig support
- ✓ safetensors library availability
- ✓ Checkpoint directory structure creation

**Output:**
```
✓ PASS: peft
✓ PASS: safetensors  
✓ PASS: structure
✓ ALL QUICK TESTS PASSED
```

## 2. Minimal Test (Recommended) ⭐

Memory-efficient test that validates save and structure without loading twice.

```bash
# Basic test (2 experts, CodeGemma-2B)
python test_checkpoint_minimal.py

# Keep checkpoint for inspection
python test_checkpoint_minimal.py --keep-checkpoint

# Test with 4 experts
python test_checkpoint_minimal.py --experts 4
```

**Tests:**
- ✓ Creates DyLoRA-MoE model
- ✓ Saves complete checkpoint (adapters + router + skill library + config)
- ✓ Verifies checkpoint structure
- ✓ Validates checkpoint size (< 500 MB)
- ✓ Checks for merged model pollution

**Output:**
```
✓ PASS: save
✓ PASS: structure  
✓ PASS: size
✓ ALL TESTS PASSED

Total checkpoint size: 28.2 MB
```

**Advantages:**
- Only loads model once (memory efficient)
- Tests actual save logic from `train.py`
- Fast enough for frequent testing
- Won't OOM on 8GB systems

## 3. Full Test (Comprehensive)

Complete validation including checkpoint reload and output consistency.

⚠️ **Warning:** Loads model twice, may OOM on systems with < 16 GB RAM.

```bash
# Full test suite
python test_checkpoint_save_load.py

# Custom configuration
python test_checkpoint_save_load.py --model google/codegemma-2b --experts 4 --keep_checkpoint
```

**Additional tests:**
- ✓ All tests from minimal version
- ✓ Loads checkpoint into fresh model instance
- ✓ Compares logits before/after save/load (tolerance 1e-5)

**Runtime:** 3-5 minutes  
**Memory:** 10-12 GB peak

## Checkpoint Structure

PEFT's `save_pretrained()` creates nested directories:

```
checkpoint/
├── peft_adapters/
│   ├── expert_0/
│   │   └── expert_0/              # PEFT nests by adapter name
│   │       ├── adapter_config.json
│   │       └── adapter_model.safetensors  (~14 MB)
│   └── expert_1/
│       └── expert_1/
│           ├── adapter_config.json
│           └── adapter_model.safetensors  (~14 MB)
├── router.safetensors             # Router state (~20 KB)
└── config.json                    # Model metadata
```

**Size expectations:**
- 2 experts: ~28 MB
- 4 experts: ~56 MB
- Router: ~20 KB

**Should NOT contain:**
- ❌ `model.safetensors` (2.5 GB - merged base model)
- ❌ `pytorch_model.bin` (2.5 GB - merged base model)

## Troubleshooting

### OOM Error During Load Test

**Symptom:** Process killed (exit code 137) during "TEST 3: Load Checkpoint"

**Solution:** Use minimal test instead:
```bash
python test_checkpoint_minimal.py  # Only loads model once
```

### Nested PEFT Directories

**Symptom:** Checkpoint has `expert_0/expert_0/adapter_model.safetensors`

**Status:** ✓ Expected behavior from PEFT's `save_pretrained()`. Tests handle both flat and nested structures automatically.

### Missing HuggingFace Token

**Problem:** Cannot load `google/codegemma-2b`

**Solution:**
```bash
export HF_TOKEN=your_token
python test_checkpoint_minimal.py
```

### Checkpoint Pollution (Large Size)

**Symptom:**
```
✗ Found merged model file: model.safetensors
Total checkpoint size: 2530.5 MB  # Should be < 100 MB!
```

**Root cause:** `TrainingArguments(save_safetensors=True)` creates merged model

**Fix in train.py:**
```python
# Ensure this is set to False
training_args = TrainingArguments(
    save_safetensors=False,  # Prevent 2.5GB merged model
    ...
)
```

### SkillLibrary Errors

**Old error:** `'SkillLibrary' object has no attribute 'state_dict'`

**Fixed:** Tests now use `skill_library.save()` and `SkillLibrary.load()` methods (not `state_dict()`).

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Checkpoints
on: [push, pull_request]

jobs:
  test-checkpoints:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: pip install -r requirements.txt
        
      - name: Quick test (environment)
        run: python test_checkpoint_quick.py
        
      - name: Minimal test (save + structure)
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: python test_checkpoint_minimal.py
```

**Recommendation:** Use `test_checkpoint_minimal.py` in CI for speed and memory efficiency.

## Usage in Training

The checkpoint save logic in `train.py` (PeftCheckpointCallback) follows the same pattern tested here:

```python
# PeftCheckpointCallback.on_save() - simplified
for expert_id in range(num_experts):
    adapter_name = f"expert_{expert_id}"
    expert_dir = os.path.join(peft_dir, adapter_name)
    
    # PEFT's native save (creates nested dirs)
    peft_model.save_pretrained(expert_dir, selected_adapters=[adapter_name])

# Extend PEFT pattern with router (not in PEFT's save_pretrained)
save_file(router.state_dict(), "router.safetensors")

# Save config
json.dump(config, open("config.json", "w"))
```

**Testing workflow:**
1. Modify checkpoint logic in `train.py`
2. Run `python test_checkpoint_minimal.py` to validate
3. If passes, run full training to verify in production

## Development Workflow

### Quick iteration
```bash
# After changing checkpoint code
python test_checkpoint_quick.py     # 5s - verify structure
python test_checkpoint_minimal.py   # 30s - verify save works
```

### Before commit
```bash
# Full validation
python test_checkpoint_minimal.py --keep-checkpoint

# Inspect checkpoint manually
ls -lh /tmp/dylora_checkpoint_minimal_*/checkpoint/
tree /tmp/dylora_checkpoint_minimal_*/checkpoint/
```

### Before release
```bash
# Test multiple configurations
python test_checkpoint_minimal.py --experts 2
python test_checkpoint_minimal.py --experts 4

# Optional: Full consistency test (if memory allows)
python test_checkpoint_save_load.py
```

## Key Design Decisions

### Why PEFT's `save_pretrained()`?
- Standard format compatible with HuggingFace ecosystem
- Handles adapter weights correctly
- Auto-detects safetensors vs PyTorch format

### Why separate router.safetensors?
- PEFT's `save_pretrained()` doesn't include router (despite `attach_router()` existing)
- We extend PEFT's pattern following naming conventions
- Router loading: `load_file("router.safetensors")` → `router.load_state_dict()`

## Test Coverage

| Component | Quick | Minimal | Full |
|-----------|-------|---------|------|
| PEFT library check | ✓ | - | - |
| Directory structure | ✓ | ✓ | ✓ |
| Save adapters | - | ✓ | ✓ |
| Save router | - | ✓ | ✓ |
| Save config | - | ✓ | ✓ |
| Load checkpoint | - | - | ✓ |
| Output consistency | - | - | ✓ |
| Size validation | - | ✓ | ✓ |
| Pollution check | - | ✓ | ✓ |

**Recommendation:** Use **minimal test** for 95% of development work. Reserve full test for final validation.
