# EvalPlus PEFT MoE Architecture Comparison

## Current Architecture (Problem)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         benchmark.py                                │
│                         (~800 lines)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────┐    ┌──────────────────────┐               │
│  │ load_base_model()  │    │ load_trained_model() │               │
│  │   (~50 lines)      │    │   (~150+ lines)      │               │
│  └────────┬───────────┘    └──────────┬───────────┘               │
│           │                           │                            │
│           │ ┌─────────────────────────┴─────────────┐             │
│           │ │  Complex PEFT Detection Logic         │             │
│           │ │  • Check for adapter_config.json      │             │
│           │ │  • Check for dylo_moe_state/          │             │
│           │ │  • Check config.json model_type       │             │
│           │ │  • Try PEFT loading                   │             │
│           │ │  • Fallback to AutoModel              │             │
│           │ │  • Fix config.json if needed          │             │
│           │ └───────────────────────────────────────┘             │
│           │                           │                            │
│           ▼                           ▼                            │
│  ┌───────────────────┐    ┌───────────────────────┐              │
│  │ AutoModelForCausalLM │ │ DyLoRA_MoE Model    │              │
│  │   (base)          │    │  • ExpertManager     │              │
│  └───────────────────┘    │  • Router            │              │
│                            │  • PEFT adapters     │              │
│  ┌────────────────────┐   └───────────────────────┘              │
│  │load_wandb_artifact()│                                          │
│  │   (~80+ lines)      │                                          │
│  │ • Download artifact │                                          │
│  │ • Extract           │                                          │
│  │ • Call load_trained │                                          │
│  └────────────────────┘                                           │
│           │                                                        │
│           ▼                                                        │
│  ┌───────────────────────────────────┐                           │
│  │    EvalPlusBenchmark              │                           │
│  │  (wrapper, doesn't actually use   │                           │
│  │   loaded models for generation!)  │                           │
│  └───────────┬───────────────────────┘                           │
│              │                                                     │
└──────────────┼─────────────────────────────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │   run_codegen()      │
    │   (from EvalPlus)    │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   make_model()       │
    │   (EvalPlus factory) │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ HuggingFaceDecoder   │
    │ • Only loads standard│
    │   HF models!         │
    │ • No PEFT support    │ ❌ PROBLEM!
    │ • No MoE routing     │
    └──────────────────────┘
               │
               ▼
    ❌ Cannot properly evaluate DyLoRA-MoE models!

ISSUES:
1. Duplicate model loading logic in benchmark.py vs EvalPlus provider
2. Complex PEFT detection duplicated across functions
3. EvalPlus HuggingFaceDecoder doesn't support loaded models
4. No reusability - loading logic locked in benchmark.py
5. No standardized interface for PEFT models
```

## Proposed Architecture (Solution)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         benchmark.py                                │
│                         (~400 lines)                                │
│                     Simplified orchestration only                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  def run_benchmarks(                                               │
│      base_model_name: str = None,                                  │
│      trained_model_path: str = None,                               │
│      wandb_artifact: str = None,                                   │
│      backend: str = "peft_moe",  # NEW                             │
│      routing_strategy: str = "router",  # NEW                      │
│      ...                                                            │
│  ):                                                                 │
│      # Delegate all loading to EvalPlus providers                  │
│      results = {}                                                   │
│      for model_config in models_to_eval:                           │
│          samples = run_codegen(                                     │
│              model=model_config["model"],                           │
│              backend=model_config["backend"],  # "hf" or "peft_moe"│
│              **model_config                                         │
│          )                                                          │
│          results[...] = evaluate(dataset, samples)                 │
│      return results                                                 │
│                                                                     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   run_codegen()       │
              │   (from EvalPlus)     │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   make_model()        │
              │   (Extended factory)  │
              └───────────┬───────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
┌────────────────────┐         ┌──────────────────────────┐
│ HuggingFaceDecoder │         │   PeftMoEDecoder (NEW)   │
│ (unchanged)        │         ├──────────────────────────┤
│ • Standard HF      │         │ All PEFT/MoE logic here  │
│   models           │         ├──────────────────────────┤
└────────────────────┘         │ def __init__:            │
         │                     │   • Detect model format  │
         │                     │   • Load W&B artifact    │
         │                     │   • Load base model      │
         │                     │   • Load PEFT adapters   │
         │                     │   • Configure routing    │
         │                     │                          │
         │                     │ def codegen():           │
         │                     │   • Select expert(s)     │
         │                     │   • Generate code        │
         │                     │   • Return completions   │
         │                     └──────────┬───────────────┘
         │                                │
         ▼                                ▼
┌────────────────────┐         ┌──────────────────────────┐
│ Standard HF Model  │         │    DyLoRA-MoE Model      │
│                    │         ├──────────────────────────┤
└────────────────────┘         │ Components:              │
                                │ • Base model (frozen)    │
                                │ • ExpertManager          │
                                │ • Router                 │
                                │ • PEFT adapters          │
                                │                          │
                                │ Routing Strategies:      │
                                │ • "router" (internal)    │
                                │ • "single:<id>"          │
                                │ • "ensemble"             │
                                │ • "best"                 │
                                └──────────────────────────┘

✅ BENEFITS:
1. Single source of truth for PEFT model loading
2. Reusable across all EvalPlus workflows
3. Clean separation: orchestration vs loading
4. Easy to extend with new routing strategies
5. Standardized interface for all model types
6. Backward compatible with existing EvalPlus
```

## Data Flow Comparison

### Current Flow (Fragmented)

```
User Command
    │
    ▼
benchmark.py CLI parsing
    │
    ├─► load_base_model() ──────────────► AutoModelForCausalLM
    │
    ├─► load_trained_model()
    │       │
    │       ├─► Check adapter_config.json
    │       ├─► Check dylo_moe_state/
    │       ├─► Try PEFT loading
    │       ├─► Fix config if needed
    │       └─► Load model ──────────────► DyLoRA_MoE
    │
    ├─► load_wandb_artifact()
    │       │
    │       ├─► Download artifact
    │       └─► call load_trained_model()
    │
    ▼
EvalPlusBenchmark.run_benchmark()
    │
    ▼
run_codegen(model_name)  ❌ Ignores loaded models!
    │
    ▼
make_model() → HuggingFaceDecoder
    │
    ▼
AutoModelForCausalLM.from_pretrained(model_name)
    │
    ▼
❌ Loads wrong model! Doesn't use PEFT model we loaded!
```

### Proposed Flow (Unified)

```
User Command
    │
    ▼
benchmark.py CLI parsing
    │
    ├─► Extract model paths and configs
    │   (no loading, just configuration)
    │
    ▼
run_codegen(
    model=path_or_name,
    backend="peft_moe",  ← NEW
    base_model=base_name,
    routing_strategy="router",
    wandb_artifact=artifact_path
)
    │
    ▼
make_model(backend="peft_moe")
    │
    ▼
PeftMoEDecoder.__init__()
    │
    ├─► detect_model_format()
    │
    ├─► load_from_wandb_artifact() if needed
    │
    ├─► Load base model
    │
    ├─► Load PEFT adapters
    │
    ├─► Configure routing strategy
    │
    ▼
PeftMoEDecoder.codegen(prompt)
    │
    ├─► Select expert(s) based on strategy
    │
    ├─► Generate with model
    │
    ▼
Return completions
    │
    ▼
✅ Correct PEFT MoE model used throughout!
```

## Code Complexity Reduction

### benchmark.py

**Before**:
```python
def load_trained_model(model_path, tokenizer, hf_token, fallback_base_model, force_device):
    """~200 lines of complex logic"""
    # Detect format
    config_path = os.path.join(model_path, "config.json")
    is_dylora_moe = False
    # ... 50 lines of detection logic ...
    
    if is_dylora_moe:
        # ... 80 lines of DyLoRA loading ...
        effective_base_model = base_model_name or fallback_base_model
        # ... fix config.json ...
        # ... try PEFT loading ...
        # ... fallback approaches ...
    elif os.path.exists(adapter_config):
        # ... 30 lines of PEFT loading ...
    else:
        # ... 40 lines of standard loading ...
```

**After**:
```python
# All moved to PeftMoEDecoder!
# benchmark.py just configures and calls

def run_benchmarks(...):
    """~50 lines of orchestration"""
    for model_config in models_to_eval:
        samples = run_codegen(
            model=model_config["model"],
            backend=model_config["backend"],
            **model_config
        )
        results[...] = evaluate(dataset, samples)
```

### Provider Implementation

**New: PeftMoEDecoder** (centralized, reusable):
```python
class PeftMoEDecoder(DecoderBase):
    def __init__(self, name, base_model, routing_strategy, wandb_artifact, ...):
        # Auto-detect format
        self.format = detect_model_format(name)
        
        # Load from W&B if needed
        if wandb_artifact:
            name = load_from_wandb_artifact(wandb_artifact)
        
        # Load based on format
        if self.format == "dylora_moe":
            self.model = self._load_dylora_moe(name, base_model)
        elif self.format == "peft_adapter":
            self.model = self._load_peft_adapter(name, base_model)
        else:
            self.model = self._load_merged(name)
        
        # Configure routing
        self.routing_strategy = routing_strategy
        self._setup_routing()
    
    def codegen(self, prompt, ...):
        # Apply routing strategy
        if self.routing_strategy.startswith("single:"):
            expert_id = int(self.routing_strategy.split(":")[1])
            self.model.expert_manager.set_active_expert(expert_id)
        # ... generate ...
```

## LOC Reduction Summary

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| benchmark.py model loading | ~350 lines | ~50 lines | **86%** |
| load_base_model | ~50 lines | Removed | **100%** |
| load_trained_model | ~200 lines | Removed | **100%** |
| load_wandb_artifact | ~100 lines | Removed | **100%** |
| **NEW** PeftMoEDecoder | 0 lines | ~400 lines | N/A (new) |
| **Total benchmark.py** | **~800 lines** | **~400 lines** | **50%** |

**Net Result**: 
- ~400 lines removed from benchmark.py
- ~400 lines added to reusable PeftMoEDecoder
- **Much cleaner architecture**
- **Reusable across projects**

## Integration Points

```
┌──────────────────────────────────────────────────────────┐
│              User Interfaces                             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────┐ │
│  │ benchmark.py   │  │ evalplus CLI   │  │ Custom    │ │
│  │ (simplified)   │  │ (extended)     │  │ Scripts   │ │
│  └────────┬───────┘  └────────┬───────┘  └─────┬─────┘ │
│           │                    │                │       │
└───────────┼────────────────────┼────────────────┼───────┘
            │                    │                │
            └────────────┬───────┴────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   EvalPlus Framework   │
            │   • run_codegen()      │
            │   • evaluate()         │
            │   • make_model()       │
            └────────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌──────────────────┐         ┌────────────────────────┐
│ Standard Backends│         │ PeftMoEDecoder (NEW)   │
│ • hf             │         │ • dylora_moe format    │
│ • vllm           │         │ • peft_adapter format  │
│ • openai         │         │ • merged format        │
│ • ...            │         │ • W&B artifacts        │
└──────────────────┘         │ • Routing strategies   │
                             └────────────────────────┘
                                         │
                                         ▼
                             ┌────────────────────────┐
                             │  Model Implementations │
                             │ • DyLoRA-MoE           │
                             │ • X-LoRA               │
                             │ • Standard PEFT        │
                             │ • Custom MoE           │
                             └────────────────────────┘
```

## Backward Compatibility

```
Old Code (Still Works):
python benchmark.py --model google/codegemma-2b
    ↓
backend="hf" (default)
    ↓
HuggingFaceDecoder (unchanged)
    ↓
✅ Works exactly as before

New Code (Enhanced):
python benchmark.py --model ./trained_model --backend peft_moe
    ↓
backend="peft_moe" (explicit)
    ↓
PeftMoEDecoder (new)
    ↓
✅ PEFT MoE support!
```

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Architecture** | Fragmented, duplicate logic | Unified, single source of truth |
| **Complexity** | ~800 lines in benchmark.py | ~400 lines orchestration |
| **Reusability** | Low (locked in benchmark.py) | High (provider pattern) |
| **Maintainability** | Difficult (scattered logic) | Easy (centralized) |
| **Extensibility** | Hard (need to modify benchmark) | Easy (add routing strategies) |
| **Testing** | Difficult (tightly coupled) | Easy (isolated providers) |
| **EvalPlus Integration** | Broken (doesn't use loaded models) | Native (proper provider) |
| **W&B Support** | Manual, in benchmark.py | Automatic, in provider |
| **Expert Selection** | None | Multiple strategies |
| **Documentation** | Scattered | Centralized |
