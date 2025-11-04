# EvalPlus PEFT MoE Integration Plan

## Executive Summary

This document proposes a comprehensive plan to extend the EvalPlus framework to support evaluation of PEFT (Parameter-Efficient Fine-Tuning) models with Mixture-of-Experts (MoE) architectures like DyLoRA-MoE. The integration will maintain backward compatibility with existing EvalPlus workflows while adding first-class support for advanced PEFT architectures.

## Current State Analysis

### EvalPlus Architecture

**Provider Pattern**:
- Abstract base: `DecoderBase` defines interface
- Concrete providers: `HuggingFaceDecoder`, `VllmDecoder`, `OpenAIDecoder`, etc.
- Factory: `make_model()` creates appropriate provider based on backend
- Each provider implements:
  - `__init__`: Model loading and configuration
  - `codegen()`: Generate code completions (returns `List[str]`)
  - `is_direct_completion()`: Prompt format detection

**Current Limitations**:
1. Only supports standard HuggingFace models via `AutoModelForCausalLM.from_pretrained()`
2. No support for PEFT adapters (LoRA, AdaLoRA, etc.)
3. No support for MoE routing mechanisms
4. Cannot load models from W&B artifacts
5. No support for multi-adapter evaluation

### DyLoRA-MoE Architecture

**Components**:
1. **Base Model**: Frozen foundation model (e.g., CodeGemma-2B @ 2.5B params)
2. **Experts**: PEFT LoRA adapters (~16M params each, ~2.5% overhead)
3. **Router**: `DynamicHybridRouter` for expert selection (dense training / sparse inference)
4. **ExpertManager**: PEFT adapter lifecycle management via `add_adapter()` / `set_adapter()`

**Generation Modes**:
- **Single-expert**: Activate one expert via `set_active_expert(i)` + standard generation
- **Router-based**: Use router to select best expert based on input hidden states
- **Multi-expert MoE**: Activate all experts + routing weights for combined inference

**Key Methods**:
```python
# Single expert selection
expert_manager.set_active_expert(expert_id)
outputs = model.generate(input_ids, ...)

# Multi-expert routing
expert_manager.activate_all_experts()
outputs = model.generate(input_ids, ...)  # Uses internal routing
```

### Current benchmark.py Issues

**Problems**:
1. **Duplicate Logic**: Complex model loading in `benchmark.py` duplicates provider functionality
2. **Tight Coupling**: Model loading mixed with benchmark orchestration
3. **Limited Reusability**: Cannot leverage model loading from other tools
4. **Maintenance**: Changes require updates in multiple places

**Current Flow**:
```
benchmark.py → load_trained_model() → complex PEFT detection → DyLoRA_MoE loading
             → load_wandb_artifact() → download + load_trained_model()
             → EvalPlusBenchmark → run_codegen() → HuggingFaceDecoder (doesn't support PEFT!)
```

## Proposed Solution

### Overview

Create a new **PeftMoEDecoder** provider that:
1. Extends EvalPlus's `DecoderBase` interface
2. Handles PEFT model loading (LoRA, QLoRA, AdaLoRA, etc.)
3. Supports MoE routing configurations
4. Integrates with W&B artifacts
5. Provides expert selection strategies

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     EvalPlus Framework                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────┐      ┌──────────────────────────┐         │
│  │ codegen.py │─────▶│  make_model() Factory    │         │
│  └────────────┘      └──────────────────────────┘         │
│                                │                           │
│                                ▼                           │
│         ┌──────────────────────┴──────────────────┐       │
│         │                                          │       │
│    ┌────▼─────┐                           ┌───────▼────┐  │
│    │ HFDecoder│                           │ PeftMoEDecoder│ (NEW)
│    └──────────┘                           └───────┬────┘  │
│         │                                         │       │
└─────────┼─────────────────────────────────────────┼───────┘
          │                                         │
          ▼                                         ▼
   ┌─────────────┐                    ┌──────────────────────┐
   │ Standard HF │                    │   PEFT MoE Models    │
   │   Models    │                    ├──────────────────────┤
   └─────────────┘                    │ • Load base model    │
                                      │ • Load PEFT adapters │
                                      │ • Configure routing  │
                                      │ • Expert selection   │
                                      │ • W&B artifacts      │
                                      └──────────────────────┘
```

## Implementation Plan

### Phase 1: Create PeftMoEDecoder Provider

**File**: `evalplus/evalplus/provider/peft_moe.py`

**Key Features**:
```python
class PeftMoEDecoder(DecoderBase):
    """
    EvalPlus provider for PEFT models with MoE support.
    
    Supports:
    - Standard PEFT models (LoRA, QLoRA, AdaLoRA)
    - MoE architectures (DyLoRA-MoE, X-LoRA)
    - W&B artifact loading
    - Expert selection strategies
    """
    
    def __init__(
        self,
        name: str,  # Model path or W&B artifact
        base_model: str = None,  # Base model name
        adapter_path: str = None,  # Path to adapters
        routing_strategy: str = "router",  # "router", "best", "ensemble", "single:<id>"
        dataset: str = "humaneval",
        wandb_artifact: str = None,  # W&B artifact path
        **kwargs
    ):
        """Initialize PEFT MoE decoder."""
        
    def _load_model(self):
        """Load PEFT model with MoE support."""
        # 1. Detect model format (DyLoRA-MoE, PEFT adapter, merged model)
        # 2. Load base model
        # 3. Load adapters
        # 4. Configure routing if MoE
        # 5. Setup expert selection strategy
        
    def _select_expert_for_prompt(self, prompt: str) -> int:
        """Select best expert for given prompt (if applicable)."""
        
    @torch.inference_mode()
    def codegen(self, prompt: str, do_sample: bool = True, 
                num_samples: int = 200) -> List[str]:
        """Generate code completions."""
        # Apply routing strategy
        # Generate with selected expert(s)
        # Return completions
```

**Routing Strategies**:
1. **"router"**: Use model's internal router (DyLoRA-MoE default)
2. **"best"**: Analyze prompt, select single best expert
3. **"ensemble"**: Generate with all experts, combine results
4. **"single:<id>"**: Force specific expert (e.g., "single:0")
5. **"round_robin"**: Cycle through experts for diversity testing

**Model Format Detection**:
```python
def detect_model_format(model_path: str) -> str:
    """
    Detect PEFT model format.
    
    Returns:
        "dylora_moe": DyLoRA-MoE with router state
        "peft_adapter": Standard PEFT adapter (adapter_config.json)
        "merged": Merged PEFT model
        "unknown": Cannot determine format
    """
    # Check for DyLoRA-MoE indicators
    if os.path.exists(os.path.join(model_path, "dylo_moe_state")):
        return "dylora_moe"
    
    # Check for PEFT adapter
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        return "peft_adapter"
    
    # Check for merged model with config
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            if config.get("model_type") == "dylora-moe":
                return "dylora_moe"
            if "base_model_name_or_path" in config:
                return "merged"
    
    return "unknown"
```

**W&B Integration**:
```python
def load_from_wandb_artifact(artifact_path: str) -> str:
    """
    Download W&B artifact and return local path.
    
    Args:
        artifact_path: Format "entity/project/artifact:version"
    
    Returns:
        Local path to downloaded artifact
    """
    import wandb
    
    # Initialize temporary run for artifact download
    with wandb.init(mode="offline") as run:
        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
    
    return artifact_dir
```

### Phase 2: Extend Provider Factory

**File**: `evalplus/evalplus/provider/__init__.py`

**Modifications**:
```python
def make_model(
    model: str,
    backend: str,
    dataset: str,
    # ... existing params ...
    
    # New PEFT MoE params
    base_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
    routing_strategy: str = "router",
    wandb_artifact: Optional[str] = None,
    is_peft_moe: bool = False,  # Auto-detect or explicit
    
    **kwargs,
) -> DecoderBase:
    """Create decoder with PEFT MoE support."""
    
    # Auto-detect PEFT MoE if not explicitly specified
    if not is_peft_moe and backend == "hf":
        # Check if model path contains PEFT indicators
        if adapter_path or wandb_artifact:
            is_peft_moe = True
        elif os.path.isdir(model):
            fmt = detect_model_format(model)
            if fmt in ["dylora_moe", "peft_adapter"]:
                is_peft_moe = True
    
    if is_peft_moe or backend == "peft_moe":
        from evalplus.provider.peft_moe import PeftMoEDecoder
        
        return PeftMoEDecoder(
            name=model,
            base_model=base_model,
            adapter_path=adapter_path,
            routing_strategy=routing_strategy,
            dataset=dataset,
            wandb_artifact=wandb_artifact,
            batch_size=batch_size,
            temperature=temperature,
            force_base_prompt=force_base_prompt,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            **kwargs,
        )
    
    # ... existing backends ...
```

### Phase 3: Refactor benchmark.py

**Simplification Strategy**:

Remove complex model loading logic and delegate to provider:

**Before** (current):
```python
# benchmark.py: ~800 lines with complex loading logic
def load_base_model(model_name, ...):
    # 50 lines of loading logic
    
def load_trained_model(model_path, ...):
    # 150+ lines of PEFT detection, format handling, etc.
    
def load_wandb_artifact(artifact_path, ...):
    # 80+ lines of download + loading
    
def run_benchmarks(models, ...):
    # Complex orchestration
```

**After** (proposed):
```python
# benchmark.py: ~400 lines, focused on orchestration
def run_benchmarks(
    base_model_name: str = None,
    trained_model_path: str = None,
    wandb_artifact: str = None,
    benchmarks: List[str] = ["humaneval"],
    backend: str = "hf",  # or "peft_moe"
    routing_strategy: str = "router",
    **kwargs
):
    """
    Run benchmarks with simplified model configuration.
    
    All model loading logic delegated to EvalPlus providers.
    """
    
    # Configure models via EvalPlus codegen
    models_to_eval = []
    
    if base_model_name:
        models_to_eval.append({
            "name": "base",
            "model": base_model_name,
            "backend": "hf"
        })
    
    if trained_model_path:
        models_to_eval.append({
            "name": "trained",
            "model": trained_model_path,
            "backend": "peft_moe",
            "base_model": base_model_name,
            "routing_strategy": routing_strategy
        })
    
    if wandb_artifact:
        models_to_eval.append({
            "name": "wandb",
            "model": wandb_artifact,
            "backend": "peft_moe",
            "base_model": base_model_name,
            "wandb_artifact": wandb_artifact,
            "routing_strategy": routing_strategy
        })
    
    # Run EvalPlus evaluation for each model
    results = {}
    for model_config in models_to_eval:
        for benchmark in benchmarks:
            samples_path = run_codegen(
                model=model_config["model"],
                dataset=benchmark,
                backend=model_config["backend"],
                **model_config,
                **kwargs
            )
            
            # Evaluate
            result = evaluate(dataset=benchmark, samples=samples_path)
            results[f"{model_config['name']}_{benchmark}"] = result
    
    return results
```

**New CLI Interface**:
```bash
# Base model only
python benchmark.py --model google/codegemma-2b --benchmarks humaneval

# PEFT model with base
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
    --backend peft_moe \
    --benchmarks humaneval

# Compare base + trained
python benchmark.py \
    --model google/codegemma-2b \
    --trained_model ./results/best_model \
    --base_model google/codegemma-2b \
    --benchmarks humaneval mbpp

# Test different routing strategies
python benchmark.py \
    --model ./results/best_model \
    --backend peft_moe \
    --routing_strategy single:0 \
    --benchmarks humaneval  # Test expert 0
    
python benchmark.py \
    --model ./results/best_model \
    --backend peft_moe \
    --routing_strategy ensemble \
    --benchmarks humaneval  # Test all experts combined
```

### Phase 4: Extended Features

#### 4.1 Expert-Specific Analysis

Add capability to evaluate individual experts:

```python
# New script: expert_analysis.py
def analyze_expert_performance(
    model_path: str,
    base_model: str,
    dataset: str = "humaneval",
    max_samples: int = None
):
    """
    Evaluate each expert individually to understand specialization.
    
    Generates reports showing:
    - Per-expert pass@1 rates
    - Task categories each expert excels at
    - Routing statistics (which expert was selected most)
    """
    
    num_experts = detect_num_experts(model_path)
    
    results = {}
    for expert_id in range(num_experts):
        # Evaluate with single expert
        result = run_codegen(
            model=model_path,
            dataset=dataset,
            backend="peft_moe",
            base_model=base_model,
            routing_strategy=f"single:{expert_id}",
            max_samples=max_samples
        )
        results[f"expert_{expert_id}"] = evaluate(dataset, result)
    
    # Compare with routing
    routing_result = run_codegen(
        model=model_path,
        dataset=dataset,
        backend="peft_moe",
        base_model=base_model,
        routing_strategy="router",
        max_samples=max_samples
    )
    results["router"] = evaluate(dataset, routing_result)
    
    # Generate comparison report
    generate_expert_analysis_report(results)
```

#### 4.2 Routing Visualization

Visualize routing decisions:

```python
def visualize_routing_patterns(
    model_path: str,
    base_model: str,
    dataset: str = "humaneval"
):
    """
    Generate routing heatmaps and statistics.
    
    Shows:
    - Expert selection distribution
    - Task characteristics vs expert selection
    - Routing confidence scores
    """
    # Instrument model to capture routing weights
    # Generate visualizations
```

#### 4.3 Ensemble Strategies

Implement multiple ensemble approaches:

```python
class EnsembleStrategy:
    """Base class for ensemble strategies."""
    
    @abstractmethod
    def combine_outputs(self, expert_outputs: List[str]) -> str:
        """Combine multiple expert outputs."""
        pass

class MajorityVotingEnsemble(EnsembleStrategy):
    """Select most common output across experts."""
    
class ConfidenceWeightedEnsemble(EnsembleStrategy):
    """Weight by generation confidence scores."""
    
class SemanticMergingEnsemble(EnsembleStrategy):
    """Merge outputs based on semantic similarity."""
```

### Phase 5: Documentation

#### 5.1 User Guide

Create `docs/PEFT_MOE_EVALUATION.md`:

```markdown
# Evaluating PEFT MoE Models with EvalPlus

## Quick Start

### Standard Evaluation
python benchmark.py --model ./trained_model --base_model google/codegemma-2b --backend peft_moe

### Expert Analysis
python expert_analysis.py --model ./trained_model --base_model google/codegemma-2b

## Configuration Options
- routing_strategy: "router", "best", "ensemble", "single:<id>"
- base_model: Foundation model name
- adapter_path: Explicit adapter path (if not in model dir)

## Model Formats Supported
- DyLoRA-MoE
- Standard PEFT adapters
- Merged PEFT models
- W&B artifacts
```

#### 5.2 API Documentation

Document PeftMoEDecoder interface and usage patterns.

#### 5.3 Examples

Provide example notebooks and scripts:
- `examples/evaluate_dylora_moe.py`
- `examples/expert_comparison.ipynb`
- `examples/routing_analysis.ipynb`

## Testing Strategy

### Unit Tests

```python
# tests/test_peft_moe_provider.py
def test_peft_moe_decoder_initialization():
    """Test provider initialization with various configs."""
    
def test_model_format_detection():
    """Test detection of DyLoRA-MoE vs PEFT adapter formats."""
    
def test_routing_strategies():
    """Test different routing strategy implementations."""
    
def test_wandb_artifact_loading():
    """Test loading models from W&B artifacts."""
```

### Integration Tests

```python
# tests/test_peft_moe_integration.py
def test_full_evaluation_pipeline():
    """Test complete evaluation with PeftMoEDecoder."""
    
def test_expert_selection():
    """Test different expert selection strategies."""
    
def test_comparison_with_base():
    """Test comparing PEFT model with base model."""
```

### Benchmark Tests

```python
# tests/benchmark_tests/test_dylora_moe_eval.py
def test_humaneval_evaluation():
    """Test HumanEval evaluation with DyLoRA-MoE."""
    
def test_mbpp_evaluation():
    """Test MBPP evaluation with DyLoRA-MoE."""
```

## Migration Guide

### For Existing DyLoRA-MoE Users

**Old approach** (complex):
```python
# Load model manually
model, tokenizer = load_trained_model(path, ...)
# Configure benchmark
benchmark = EvalPlusBenchmark(...)
# Run evaluation
results = benchmark.run_benchmark(model, ...)
```

**New approach** (simplified):
```python
# Everything handled by provider
python benchmark.py \
    --model ./trained_model \
    --base_model google/codegemma-2b \
    --backend peft_moe \
    --benchmarks humaneval mbpp
```

### For EvalPlus Users

No breaking changes - existing workflows continue to work. New backend is opt-in via `--backend peft_moe`.

## Implementation Timeline

| Phase | Tasks | Estimated Effort | Dependencies |
|-------|-------|------------------|--------------|
| **Phase 1** | Create PeftMoEDecoder | 3-4 days | None |
| | - Core provider implementation | 2 days | |
| | - W&B integration | 1 day | |
| | - Routing strategies | 1 day | |
| **Phase 2** | Extend factory | 1 day | Phase 1 |
| **Phase 3** | Refactor benchmark.py | 2-3 days | Phases 1-2 |
| | - Simplify model loading | 1 day | |
| | - Update CLI | 1 day | |
| | - Testing | 1 day | |
| **Phase 4** | Extended features | 2-3 days | Phases 1-3 |
| | - Expert analysis | 1 day | |
| | - Routing visualization | 1 day | |
| | - Ensemble strategies | 1 day | |
| **Phase 5** | Documentation | 2 days | All phases |
| **Total** | | **10-13 days** | |

## Success Criteria

1. **Functional**:
   - [ ] PeftMoEDecoder loads DyLoRA-MoE models correctly
   - [ ] All routing strategies work as expected
   - [ ] W&B artifact loading functions properly
   - [ ] Evaluation results match expected accuracy

2. **Quality**:
   - [ ] Code coverage >80% for new components
   - [ ] All tests pass
   - [ ] Documentation complete and clear
   - [ ] No performance regression

3. **Usability**:
   - [ ] CLI is intuitive and well-documented
   - [ ] Error messages are helpful
   - [ ] Examples work out-of-box
   - [ ] Migration path is clear

## Benefits

### For DyLoRA-MoE Development:
- **Simplified Evaluation**: One-command evaluation instead of complex setup
- **Standardized Metrics**: Use industry-standard EvalPlus metrics
- **Better Comparisons**: Easy comparison with base models and other approaches
- **Expert Analysis**: Understand individual expert performance
- **Routing Insights**: Visualize and analyze routing decisions

### For EvalPlus Ecosystem:
- **Extended Capabilities**: Support cutting-edge PEFT MoE architectures
- **Model Flexibility**: Evaluate not just full models but adapter-based approaches
- **Research Tools**: Enable new research on expert specialization and routing
- **Community Value**: Provide tools others can use for PEFT model evaluation

### For Both:
- **Cleaner Architecture**: Separation of concerns between evaluation and model loading
- **Maintainability**: Centralized model loading logic
- **Extensibility**: Easy to add new routing strategies and model formats
- **Reusability**: Provider can be used in other EvalPlus workflows

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing workflows | High | Low | Extensive testing, backward compatibility |
| Performance issues with multi-expert | Medium | Medium | Profiling, caching, lazy loading |
| Complex routing logic | Medium | Medium | Start simple, iterate based on needs |
| W&B dependency | Low | Low | Make W&B optional, graceful degradation |
| PEFT version compatibility | Medium | Medium | Pin versions, test across PEFT versions |

## Future Enhancements

1. **Multi-backend routing**: Support routing across different backend types
2. **Dynamic adapter loading**: Load/unload adapters on-demand for memory efficiency
3. **Distributed evaluation**: Evaluate large MoE models across multiple GPUs
4. **Adaptive routing**: Learn optimal routing strategies from evaluation results
5. **Cross-model ensembles**: Combine outputs from different model families

## Conclusion

This integration plan provides a comprehensive approach to extending EvalPlus for PEFT MoE models while maintaining the simplicity and reliability of the existing framework. The phased approach allows for incremental development and testing, with each phase building on the previous ones.

The proposed `PeftMoEDecoder` provider cleanly encapsulates all PEFT-specific logic, making it easy to maintain and extend. The refactored `benchmark.py` becomes much simpler and more focused on orchestration rather than model loading details.

This design positions DyLoRA-MoE for easy, standardized evaluation while contributing valuable capabilities back to the EvalPlus ecosystem.
