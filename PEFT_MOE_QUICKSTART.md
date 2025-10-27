# PEFT MoE Routing - Quick Start Guide

This guide helps you get started with implementing MoE routing support in PEFT and integrating it with DyLoRA.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Editable install of PEFT: `pip install -e ./peft/`
- Familiarity with LoRA and DyLoRA architecture

## Implementation Order

Follow these steps in order for a smooth implementation:

### Week 1: PEFT Core (9-10 hours)

#### Day 1-2: LoraLayer Modifications (3-4 hours)
```bash
# 1. Open the file
code peft/src/peft/tuners/lora/layer.py

# 2. Navigate to line ~779 (Linear.forward method)
# 3. Add routing_weights support after adapter_names extraction
# 4. Implement weighted combination loop
# 5. Run tests: pytest peft/tests/test_lora.py -v
```

**Key Code Addition:**
```python
routing_weights = kwargs.pop("routing_weights", None)

if routing_weights is not None:
    # Weighted combination of all active adapters
    result = self.base_layer(x, *args, **kwargs)
    for i, adapter_name in enumerate(self.active_adapters):
        weight = routing_weights[..., i:i+1]
        lora_out = lora_B(lora_A(dropout(x))) * scaling * weight
        result = result + lora_out
    return result
```

#### Day 2-3: MoELoraConfig (2 hours)
```bash
# 1. Open the file
code peft/src/peft/tuners/lora/config.py

# 2. Add MoELoraConfig after LoraConfig class
# 3. Add to __init__.py exports
# 4. Test serialization: python -c "from peft import MoELoraConfig; print(MoELoraConfig())"
```

#### Day 3-4: Router Management (4 hours)
```bash
# 1. Open the file
code peft/src/peft/tuners/lora/model.py

# 2. Add router attribute to __init__
# 3. Implement attach_router() and detach_router()
# 4. Update state_dict methods
# 5. Test: python -c "from peft import LoraModel; # test instantiation"
```

### Week 2: DyLoRA Integration (8 hours)

#### Day 5-6: Forward Pass Refactor (5 hours)
```bash
# 1. Backup current model.py
cp dylo_moe/model.py dylo_moe/model.py.backup

# 2. Open the file
code dylo_moe/model.py

# 3. Remove expert loop (lines 224-245)
# 4. Add single-pass routing logic
# 5. Test forward pass: python -c "from dylo_moe.model import DyLoRA_MoE; # test"
```

**Before:**
```python
for i in range(self.router.num_experts):
    self.expert_manager.set_active_expert(i)
    expert_out = self.foundation_model(input_ids, ...)
    weighted_logits = expert_out.logits * expert_weights[:, i]
    logits = logits + weighted_logits
```

**After:**
```python
routing_weights = self.router(hidden_states)
outputs = self.foundation_model(
    input_ids,
    routing_weights=routing_weights,  # NEW!
)
logits = outputs.logits
```

#### Day 7: ExpertManager Updates (2 hours)
```bash
# 1. Open the file
code dylo_moe/expert.py

# 2. Add activate_all_experts() method
# 3. Update adapter tracking
# 4. Test: python -c "from dylo_moe.expert import ExpertManager; # test"
```

#### Day 8: Integration Testing (1 hour)
```bash
# Quick validation test
python -c "
from dylo_moe.model import DyLoRA_MoE
import torch

model = DyLoRA_MoE('google/codegemma-2b', num_experts=4, token='hf_...')
input_ids = torch.randint(0, 1000, (2, 10))
output = model(input_ids)
print(f'Output shape: {output.logits.shape}')
print('✅ Forward pass successful!')
"
```

### Week 3: Testing & Validation (16 hours)

#### Day 9-10: Unit Tests (6 hours)
```bash
# Create test file
touch peft/tests/test_moe_routing.py

# Write tests
code peft/tests/test_moe_routing.py

# Run tests
pytest peft/tests/test_moe_routing.py -v --cov=peft.tuners.lora
```

**Essential Tests:**
1. `test_routing_weights_forward_basic` - Basic weighted combination
2. `test_routing_weights_gradients` - Gradient flow
3. `test_backward_compatibility_no_routing` - Works without routing_weights
4. `test_multiple_experts_2_4_8` - Various expert counts

#### Day 11-12: Integration Tests (6 hours)
```bash
# Create test file
touch tests/test_dylora_moe_integration.py

# Write tests
code tests/test_dylora_moe_integration.py

# Run tests
pytest tests/test_dylora_moe_integration.py -v -s
```

**Essential Tests:**
1. `test_single_vs_multi_pass_output_equivalence` - Outputs match
2. `test_training_convergence_small_dataset` - Can train
3. `test_forward_pass_speedup` - Performance improvement

#### Day 13-14: Validation Script (4 hours)
```bash
# Create validation script
touch validate_moe_modifications.py

# Make it executable
chmod +x validate_moe_modifications.py

# Run validation
python validate_moe_modifications.py --checkpoint path/to/checkpoint
```

## Verification Checklist

After each phase, verify:

### ✅ Phase 1 Complete When:
- [ ] `routing_weights` kwarg accepted in `Linear.forward()`
- [ ] Weighted combination produces correct output shape
- [ ] Gradients flow through routing_weights
- [ ] Existing LoRA tests still pass
- [ ] MoELoraConfig can be instantiated
- [ ] Router can be attached/detached

**Test Command:**
```bash
pytest peft/tests/test_lora.py peft/tests/test_moe_routing.py -v
```

### ✅ Phase 2 Complete When:
- [ ] DyLoRA forward pass uses single base model pass
- [ ] Training loss decreases normally
- [ ] Routing weights are being learned
- [ ] Memory usage acceptable (<10% increase)
- [ ] Speed improvement visible (>1.5x for 4 experts)

**Test Command:**
```bash
# Quick training test
python train.py --num_experts 4 --training_subset 10 --eval_subset 20 --num_epochs 3 --bf16
```

### ✅ Phase 3 Complete When:
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Validation script confirms output equivalence
- [ ] Benchmarks show expected improvements
- [ ] No memory leaks detected

**Test Command:**
```bash
pytest tests/ peft/tests/test_moe_routing.py -v --cov
python validate_moe_modifications.py --benchmark
```

## Common Issues & Solutions

### Issue 1: Routing weights gradient is None
**Cause:** Router not connected to computational graph
**Solution:** Ensure router is called inside forward pass, not in no_grad context

### Issue 2: Shape mismatch errors
**Cause:** routing_weights shape doesn't match expected [batch, seq, num_experts]
**Solution:** Check routing_weights.shape before passing to forward

### Issue 3: Memory spike during forward pass
**Cause:** All expert outputs computed simultaneously
**Solution:** This is expected, but <10% increase should be acceptable

### Issue 4: Slower than expected
**Cause:** Base model pass still dominates
**Solution:** For small models, speedup is modest. Test with larger models.

### Issue 5: Training doesn't converge
**Cause:** Routing weights initialized poorly
**Solution:** Initialize router with small weights, use higher learning rate for router

## Performance Expectations

### Speedup by Number of Experts
| Experts | Expected Speedup | Actual Speedup (measured) |
|---------|------------------|---------------------------|
| 2       | 1.5x - 1.8x     | TBD after implementation  |
| 4       | 2.5x - 3.5x     | TBD after implementation  |
| 8       | 3.5x - 5.0x     | TBD after implementation  |

### Memory Usage
| Model Size | Experts | Old Peak | New Peak | Change |
|------------|---------|----------|----------|--------|
| 2B params  | 4       | TBD      | TBD      | TBD    |
| 7B params  | 4       | TBD      | TBD      | TBD    |

## Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Routing Weights
```python
# After forward pass
print(f"Routing weights shape: {model.last_routing_weights.shape}")
print(f"Routing weights range: [{model.last_routing_weights.min():.3f}, {model.last_routing_weights.max():.3f}]")
print(f"Expert usage: {model.last_routing_weights.mean(dim=[0,1])}")
```

### Profile Memory
```python
import torch
torch.cuda.reset_peak_memory_stats()
output = model(input_ids)
peak_memory = torch.cuda.max_memory_allocated() / 1024**3
print(f"Peak memory: {peak_memory:.2f} GB")
```

### Profile Speed
```python
import time
start = time.time()
for _ in range(100):
    output = model(input_ids)
avg_time = (time.time() - start) / 100
print(f"Average forward pass: {avg_time*1000:.2f} ms")
```

## Next Steps After Implementation

1. **Benchmark thoroughly** - Document all improvements
2. **Write blog post** - Share learnings with community
3. **Submit PEFT PR** - Contribute back to open source
4. **Update DyLoRA docs** - Help users migrate
5. **Experiment with variants** - Try per-layer routing, sparse routing, etc.

## Resources

- **Implementation Plan:** `PEFT_MOE_IMPLEMENTATION_PLAN.md`
- **Detailed TODOs:** `PEFT_MOE_TODOS.md`
- **PEFT Docs:** https://huggingface.co/docs/peft
- **DyLoRA Docs:** `README.md`

## Getting Help

If you encounter issues:

1. Check the "Common Issues" section above
2. Review the implementation plan for design rationale
3. Look at PEFT tests for usage examples
4. Open an issue in the repository with:
   - Error message and stack trace
   - Minimal code to reproduce
   - Environment details (Python version, PyTorch version, etc.)

## Success Criteria

You'll know the implementation is successful when:

- ✅ All tests pass
- ✅ Training converges normally
- ✅ Forward pass is 2-3x faster with 4 experts
- ✅ Memory usage increases <10%
- ✅ Output quality unchanged
- ✅ Code is clean and documented

Good luck! 🚀
