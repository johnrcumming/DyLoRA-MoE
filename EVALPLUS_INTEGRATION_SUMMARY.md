# EvalPlus PEFT MoE Integration - Quick Reference

> **Full details**: See [EVALPLUS_PEFT_MOE_INTEGRATION_PLAN.md](./EVALPLUS_PEFT_MOE_INTEGRATION_PLAN.md)

## Problem Statement

**Current Limitation**: EvalPlus doesn't support evaluating PEFT models with MoE architectures like DyLoRA-MoE.

**Impact**: 
- Complex, brittle model loading logic scattered across benchmark.py
- Cannot leverage EvalPlus's standardized evaluation for PEFT models
- Difficult to compare PEFT models with base models
- No support for W&B artifact-based evaluation

## Proposed Solution

Create **PeftMoEDecoder** - a new EvalPlus provider that understands PEFT models with MoE routing.

### Architecture

```
EvalPlus codegen → make_model() → PeftMoEDecoder → DyLoRA-MoE Model
                                        ↓
                                  Load adapters
                                  Configure routing
                                  Handle W&B artifacts
```

## Key Components

### 1. PeftMoEDecoder Provider (`evalplus/evalplus/provider/peft_moe.py`)

New provider extending `DecoderBase`:
- Loads PEFT models (LoRA, QLoRA, AdaLoRA, etc.)
- Detects DyLoRA-MoE format automatically
- Supports W&B artifact loading
- Implements routing strategies
- Generates code completions with expert selection

**Routing Strategies**:
- `"router"`: Use model's internal router (DyLoRA-MoE default)
- `"single:<id>"`: Force specific expert (e.g., "single:0")
- `"best"`: Analyze prompt, select best expert
- `"ensemble"`: Generate with all experts, combine results
- `"round_robin"`: Cycle through experts for diversity

### 2. Extended Factory (`evalplus/evalplus/provider/__init__.py`)

Add `peft_moe` backend to `make_model()`:
```python
def make_model(
    model: str,
    backend: str = "peft_moe",  # NEW
    base_model: str = None,     # NEW
    adapter_path: str = None,   # NEW
    routing_strategy: str = "router",  # NEW
    wandb_artifact: str = None,  # NEW
    ...
):
```

### 3. Simplified benchmark.py

**Before**: ~800 lines with complex loading logic
**After**: ~400 lines, delegates to providers

**New CLI**:
```bash
# Evaluate PEFT model with routing
python benchmark.py \
    --model ./results/best_model \
    --base_model google/codegemma-2b \
    --backend peft_moe \
    --routing_strategy router \
    --benchmarks humaneval mbpp

# W&B artifact
python benchmark.py \
    --wandb_artifact "user/project/model:v0" \
    --base_model google/codegemma-2b \
    --backend peft_moe

# Test specific expert
python benchmark.py \
    --model ./results/best_model \
    --backend peft_moe \
    --routing_strategy single:0
```

## Implementation Phases

| Phase | What | Duration |
|-------|------|----------|
| **Phase 1** | Create PeftMoEDecoder | 3-4 days |
| **Phase 2** | Extend make_model() | 1 day |
| **Phase 3** | Refactor benchmark.py | 2-3 days |
| **Phase 4** | Extended features (expert analysis, visualization) | 2-3 days |
| **Phase 5** | Documentation | 2 days |
| **Total** | | **10-13 days** |

## Key Features

### Model Format Detection
Automatically detects:
- DyLoRA-MoE format (with router state)
- PEFT adapter format (adapter_config.json)
- Merged PEFT models
- W&B artifacts

### Expert Analysis Tools
```python
# New: expert_analysis.py
python expert_analysis.py \
    --model ./trained_model \
    --base_model google/codegemma-2b

# Output:
# - Per-expert pass@1 rates
# - Task categories each expert excels at
# - Routing statistics
```

### Routing Visualization
```python
# New: routing_analysis.py
python routing_analysis.py \
    --model ./trained_model \
    --dataset humaneval

# Output:
# - Expert selection heatmaps
# - Routing confidence scores
# - Task-to-expert mapping
```

## Benefits

### For DyLoRA-MoE:
✅ One-command evaluation instead of complex setup
✅ Standardized EvalPlus metrics
✅ Easy comparison with base models
✅ Expert performance analysis
✅ Routing insights and visualization

### For EvalPlus:
✅ Support cutting-edge PEFT MoE architectures
✅ Enable research on expert specialization
✅ Extend to broader model types
✅ Community contribution

### Architecture:
✅ Clean separation of concerns
✅ Centralized model loading
✅ Easy to extend with new strategies
✅ Backward compatible

## Example Usage

### Basic Evaluation
```python
from evalplus.codegen import run_codegen
from evalplus.evaluate import evaluate

# Generate with PEFT MoE model
samples = run_codegen(
    model="./trained_model",
    backend="peft_moe",
    base_model="google/codegemma-2b",
    dataset="humaneval",
    routing_strategy="router"
)

# Evaluate
results = evaluate(dataset="humaneval", samples=samples)
print(f"Pass@1: {results['pass@1']}")
```

### Expert Comparison
```python
# Compare all routing strategies
strategies = ["router", "single:0", "single:1", "single:2", "ensemble"]

for strategy in strategies:
    samples = run_codegen(
        model="./trained_model",
        backend="peft_moe",
        routing_strategy=strategy,
        dataset="humaneval"
    )
    results = evaluate(dataset="humaneval", samples=samples)
    print(f"{strategy}: {results['pass@1']:.2f}%")
```

## Integration with Current Workflow

### Old Workflow (Current)
```python
# Complex manual loading
model, tokenizer = load_trained_model(path, ...)
model = handle_dylora_format(model, ...)
model = load_wandb_artifact_if_needed(...)

# Manual benchmark setup
benchmark = EvalPlusBenchmark(tokenizer, model_name=...)
results = benchmark.run_benchmark(model, ...)
```

### New Workflow (Proposed)
```bash
# One command - provider handles everything
python benchmark.py \
    --model ./trained_model \
    --base_model google/codegemma-2b \
    --backend peft_moe \
    --benchmarks humaneval mbpp
```

## Migration Path

**Step 1**: Implement PeftMoEDecoder (Phase 1)
- No impact on existing code
- New provider is opt-in

**Step 2**: Extend factory (Phase 2)
- Backward compatible
- Existing backends unchanged

**Step 3**: Refactor benchmark.py (Phase 3)
- Keep old functions as deprecated
- New CLI interface is cleaner
- Gradual migration

**Step 4**: Update documentation (Phase 5)
- Migration guide
- Examples
- Best practices

## Next Steps

1. **Review & Approve**: Review this plan and [full plan](./EVALPLUS_PEFT_MOE_INTEGRATION_PLAN.md)
2. **Phase 1 Implementation**: Create PeftMoEDecoder
3. **Testing**: Unit and integration tests
4. **Phase 2-3**: Factory + benchmark refactor
5. **Documentation**: User guide and examples
6. **Release**: Integrate into main workflow

## Questions to Address

1. **Routing Strategy Defaults**: Which routing strategy should be default?
   - Recommendation: `"router"` for multi-expert, `"single:0"` for single expert

2. **W&B Integration**: Required or optional dependency?
   - Recommendation: Optional - graceful degradation if not available

3. **Backward Compatibility**: Keep old benchmark.py functions?
   - Recommendation: Deprecate gradually over 2-3 releases

4. **Performance**: Memory optimization for multi-expert?
   - Recommendation: Lazy loading, caching, profiling in Phase 4

## References

- **Full Plan**: [EVALPLUS_PEFT_MOE_INTEGRATION_PLAN.md](./EVALPLUS_PEFT_MOE_INTEGRATION_PLAN.md)
- **EvalPlus Repo**: [evalplus/evalplus](https://github.com/evalplus/evalplus)
- **PEFT Docs**: [huggingface.co/docs/peft](https://huggingface.co/docs/peft)
- **DyLoRA-MoE Arch**: [.github/copilot-instructions.md](./.github/copilot-instructions.md)
