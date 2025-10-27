# PEFT MoE Implementation Plan

## Executive Summary

This document outlines the modifications needed to integrate MoE (Mixture-of-Experts) routing directly into the PEFT library, eliminating the current inefficiency of multiple forward passes through the base model.

**Current Problem:** DyLoRA-MoE performs N forward passes (one per expert) to combine expert outputs, making training N times slower than necessary.

**Solution:** Extend PEFT's `LoraLayer` to accept continuous routing weights and combine all expert adapters in a single forward pass through the base model.

## Architecture Overview

### Current DyLoRA-MoE Flow
```
Input → Base Model (Pass 1) → Expert 0 → Weighted Output 0
Input → Base Model (Pass 2) → Expert 1 → Weighted Output 1  ❌ Inefficient
Input → Base Model (Pass 3) → Expert 2 → Weighted Output 2
...
Combined Outputs → Loss
```

### Proposed PEFT-Native MoE Flow
```
Input → Base Model (Single Pass) → Hidden States → Router → Routing Weights
                                  ↓
        All Expert Adapters Applied with Weights in Parallel
                                  ↓
                        Combined Output → Loss
```

## Implementation Components

### 1. LoraLayer Modifications (`peft/src/peft/tuners/lora/layer.py`)

**Location:** ~Line 779 in `Linear.forward()`

**Changes:**
- Add `routing_weights` as optional kwarg
- When `routing_weights` is provided, apply all active adapters with continuous weights
- Maintain backward compatibility (routing_weights=None uses existing logic)

**New Forward Pass Logic:**
```python
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    routing_weights = kwargs.pop("routing_weights", None)  # NEW
    
    if routing_weights is not None:
        # MoE mode: weighted combination of all active adapters
        result = self.base_layer(x, *args, **kwargs)
        
        for i, adapter_name in enumerate(self.active_adapters):
            weight = routing_weights[..., i:i+1]  # [batch, seq, 1]
            lora_out = lora_B(lora_A(dropout(x))) * scaling * weight
            result = result + lora_out
        return result
    else:
        # Existing logic unchanged
        ...
```

**Key Benefits:**
- Single base model forward pass
- All adapters applied in same computational graph
- Gradients flow naturally through router + adapters

### 2. MoELoraConfig (`peft/src/peft/tuners/lora/config.py`)

**New Configuration Class:**
```python
@dataclass
class MoELoraConfig(LoraConfig):
    """LoRA configuration with MoE routing support"""
    
    use_moe_routing: bool = field(
        default=False, 
        metadata={"help": "Enable MoE routing for multi-expert inference"}
    )
    
    router_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Input size for router (usually model hidden_size)"}
    )
    
    num_experts: int = field(
        default=1,
        metadata={"help": "Number of expert adapters"}
    )
    
    top_k_experts: int = field(
        default=1,
        metadata={"help": "Number of experts to activate (sparse routing)"}
    )
    
    router_aux_loss_coef: float = field(
        default=0.01,
        metadata={"help": "Coefficient for load balancing auxiliary loss"}
    )
    
    router_temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for routing softmax"}
    )
```

### 3. Router Management in LoraModel (`peft/src/peft/tuners/lora/model.py`)

**New Methods:**
```python
class LoraModel(BaseTuner):
    def __init__(self, model, config, adapter_name, ...):
        super().__init__(...)
        self.router = None  # Optional router module
        self.moe_config = None
    
    def attach_router(self, router_module: nn.Module, moe_config: Optional[MoELoraConfig] = None):
        """
        Attach a router that computes routing weights from hidden states.
        
        Args:
            router_module: nn.Module that takes hidden states and returns routing weights
            moe_config: MoE configuration (optional)
        """
        self.router = router_module
        self.moe_config = moe_config
        
    def detach_router(self):
        """Remove router for standard LoRA operation"""
        self.router = None
        self.moe_config = None
    
    def get_routing_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute routing weights from hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            routing_weights: [batch, seq_len, num_experts]
        """
        if self.router is None:
            raise ValueError("No router attached. Call attach_router() first.")
        return self.router(hidden_states)
```

**Integration Points:**
- Router is optional - only used when explicitly attached
- Router state saved/loaded with model checkpoints
- Compatible with existing PEFT saving/loading infrastructure

### 4. DyLoRA_MoE Integration (`dylo_moe/model.py`)

**Simplified Forward Pass:**
```python
def forward(self, input_ids, attention_mask=None, labels=None, expert_id=None):
    # Training mode: single expert
    if expert_id is not None:
        self.expert_manager.set_active_expert(expert_id)
        outputs = self.foundation_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
    # Single expert case
    elif self.expert_manager.num_experts == 1:
        outputs = self.foundation_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
    # Multi-expert MoE routing (NEW - SINGLE PASS)
    else:
        # Get hidden states for routing
        outputs = self.foundation_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        
        # Compute routing weights
        routing_weights = self.router(hidden_states)  # [batch, seq, num_experts]
        
        # Single forward pass with routing weights
        # PEFT layers will automatically apply weighted combination
        outputs = self.foundation_model(
            input_ids,
            attention_mask=attention_mask,
            routing_weights=routing_weights,  # NEW kwarg passed to LoraLayers
            use_cache=False,
        )
        logits = outputs.logits
        
        # Store for monitoring
        self.last_routing_weights = routing_weights.detach()
    
    # Loss computation (unchanged)
    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
        
        # Load balancing loss (unchanged)
        if self.training and self.last_routing_weights is not None:
            balance_loss = self.compute_load_balancing_loss(self.last_routing_weights)
            loss = lm_loss + self.balance_coefficient * balance_loss
            self.last_lm_loss = lm_loss.detach()
            self.last_balance_loss = balance_loss.detach()
        else:
            loss = lm_loss
            self.last_lm_loss = lm_loss.detach()
            self.last_balance_loss = None
    
    from transformers.modeling_outputs import CausalLMOutputWithPast
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=None,
        hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
    )
```

**Key Changes:**
- Remove the expert loop (lines 220-245 in current implementation)
- Add `routing_weights` kwarg to foundation_model call
- PEFT handles weighted combination internally

## Implementation Steps

### Phase 1: PEFT Core Modifications

#### Step 1.1: Modify LoraLayer.forward()
**File:** `peft/src/peft/tuners/lora/layer.py`
**Lines:** ~779-820 (Linear class forward method)

**Changes:**
1. Extract `routing_weights` from kwargs
2. Add MoE routing branch before existing logic
3. Apply weighted combination of all active adapters
4. Maintain dtype consistency

**Testing:**
- Verify backward compatibility (routing_weights=None)
- Test with routing_weights of various shapes
- Confirm gradients flow through routing_weights

#### Step 1.2: Create MoELoraConfig
**File:** `peft/src/peft/tuners/lora/config.py`
**Location:** After LoraConfig class definition

**Changes:**
1. Define MoELoraConfig dataclass extending LoraConfig
2. Add MoE-specific fields with defaults
3. Add validation in __post_init__ if needed

**Testing:**
- Verify config serialization/deserialization
- Test compatibility with existing LoraConfig

#### Step 1.3: Add Router Management to LoraModel
**File:** `peft/src/peft/tuners/lora/model.py`
**Location:** LoraModel class

**Changes:**
1. Add router attribute to __init__
2. Implement attach_router() method
3. Implement detach_router() method
4. Add router to state_dict saving/loading

**Testing:**
- Verify router can be attached/detached
- Test router state persistence across save/load

### Phase 2: DyLoRA Integration

#### Step 2.1: Update DyLoRA_MoE Forward Pass
**File:** `dylo_moe/model.py`
**Lines:** ~180-250

**Changes:**
1. Remove expert loop (lines 224-245)
2. Add single forward pass with routing_weights
3. Update monitoring/logging to track single pass

**Testing:**
- Compare output logits before/after (should match)
- Verify training loss convergence
- Benchmark forward pass time (should be ~N times faster)

#### Step 2.2: Update ExpertManager
**File:** `dylo_moe/expert.py`

**Changes:**
1. Add method to set all experts as active simultaneously
2. Update expert tracking for MoE mode

**Testing:**
- Verify all experts can be active at once
- Test adapter switching still works for single-expert training

### Phase 3: Testing & Validation

#### Step 3.1: Unit Tests
**New File:** `tests/test_moe_routing.py`

**Test Cases:**
1. `test_routing_weights_forward_pass()` - Verify weighted combination math
2. `test_routing_weights_gradients()` - Check gradient flow
3. `test_backward_compatibility()` - Ensure routing_weights=None works
4. `test_multiple_experts()` - Test with 2, 4, 8 experts
5. `test_routing_weights_shapes()` - Various batch/seq_len combinations

#### Step 3.2: Integration Tests
**New File:** `tests/test_dylora_moe_integration.py`

**Test Cases:**
1. `test_single_vs_multi_pass_equivalence()` - Output should match
2. `test_training_convergence()` - Small dataset training
3. `test_memory_efficiency()` - Compare memory usage
4. `test_speed_improvement()` - Benchmark forward pass time

#### Step 3.3: Validation Script
**New File:** `validate_moe_modifications.py`

**Validation Steps:**
1. Load existing DyLoRA checkpoint
2. Run inference with old approach
3. Run inference with new approach
4. Compare outputs (should be nearly identical)
5. Report speedup and memory savings

## Migration Path

### For Existing DyLoRA Users

**Option A: Keep Current Approach (No Changes Needed)**
- Current implementation will continue to work
- No migration required
- Performance unchanged

**Option B: Upgrade to MoE-Native PEFT (Recommended)**

1. **Update PEFT:**
   ```bash
   cd peft/
   git pull origin main  # After changes are merged
   pip install -e .
   ```

2. **Minimal Code Changes:**
   ```python
   # OLD: Multiple forward passes (lines 224-245 in model.py)
   for i in range(self.router.num_experts):
       self.expert_manager.set_active_expert(i)
       expert_out = self.foundation_model(input_ids, ...)
       weighted_logits = expert_out.logits * expert_weights[:, i]
       logits = logits + weighted_logits
   
   # NEW: Single forward pass
   routing_weights = self.router(hidden_states)
   outputs = self.foundation_model(
       input_ids,
       routing_weights=routing_weights,  # New kwarg
   )
   logits = outputs.logits
   ```

3. **Verify Results:**
   - Run `validate_moe_modifications.py`
   - Check training metrics match expected values
   - Benchmark performance improvement

### Backward Compatibility Guarantees

1. **Existing LoRA models:** Continue to work without changes
2. **Single-expert DyLoRA:** No code changes needed
3. **Multi-expert DyLoRA:** Can opt-in to new routing by passing `routing_weights` kwarg

## Performance Expectations

### Speed Improvements
- **Current:** N forward passes (N = num_experts)
- **After:** 1 base forward pass + lightweight adapter application
- **Expected Speedup:** 2-5x for 4 experts (base pass dominates compute)

### Memory Usage
- **Current:** Peak memory during sequential expert evaluation
- **After:** Slightly higher peak (all adapter outputs in memory simultaneously)
- **Net Change:** +5-10% peak memory, but much faster so lower total memory-time

### Training Time
- **Small Models (CodeGemma-2B):** 40-50% reduction in training time
- **Large Models (7B+):** 60-70% reduction (base pass is bottleneck)

## Risk Assessment

### Low Risk Changes
✅ LoraLayer.forward() modification (new optional kwarg)
✅ MoELoraConfig addition (new class, doesn't affect existing)
✅ Router management methods (optional features)

### Medium Risk Changes
⚠️ DyLoRA forward pass refactor (significant logic change)
- **Mitigation:** Comprehensive testing, validation script, gradual rollout

### High Risk Changes
❌ None - all changes are additive and backward compatible

## Success Criteria

### Must Have
- [ ] LoraLayer accepts routing_weights kwarg
- [ ] Single forward pass produces correct outputs
- [ ] Gradients flow through routing_weights
- [ ] Backward compatible with existing PEFT usage
- [ ] DyLoRA training converges to same loss

### Should Have
- [ ] 2x+ speedup in multi-expert forward pass
- [ ] Memory usage increase <10%
- [ ] Comprehensive test coverage
- [ ] Documentation and examples

### Nice to Have
- [ ] Support for other adapter types (LoHa, LoKr)
- [ ] Per-layer routing (different weights per layer)
- [ ] Router checkpointing utilities
- [ ] Visualization tools for routing patterns

## Timeline Estimate

- **Week 1:** PEFT core modifications (Steps 1.1-1.3)
- **Week 2:** DyLoRA integration (Steps 2.1-2.2)
- **Week 3:** Testing and validation (Steps 3.1-3.3)
- **Week 4:** Documentation, examples, PR preparation

**Total:** ~4 weeks for complete implementation and testing

## Open Questions

1. **Routing granularity:** Should routing weights be per-token or per-sequence?
   - **Current approach:** Per-token (max flexibility)
   - **Alternative:** Per-sequence (simpler, faster)
   - **Decision:** Start with per-token, optimize later if needed

2. **Router state management:** How to save/load router with PEFT checkpoints?
   - **Option A:** Separate router checkpoint
   - **Option B:** Include in PEFT state_dict
   - **Decision:** Option B for simplicity

3. **Multi-layer routing:** Should each layer have its own router?
   - **Current approach:** Single shared router
   - **Alternative:** Per-layer routers for fine-grained control
   - **Decision:** Start with shared, add per-layer as advanced feature

## References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Mixtral MoE Architecture](https://arxiv.org/abs/2401.04088)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)

## Appendix: Code Diffs

### A. LoraLayer.forward() Modification

```python
# In peft/src/peft/tuners/lora/layer.py, Line ~779

# BEFORE:
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}

    if self.disable_adapters:
        # ... existing logic

# AFTER:
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)
    routing_weights = kwargs.pop("routing_weights", None)  # NEW
    variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}

    if self.disable_adapters:
        # ... existing logic
    elif routing_weights is not None:  # NEW BRANCH
        # MoE routing mode with continuous weights
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        
        lora_A_keys = self.lora_A.keys()
        for i, active_adapter in enumerate(self.active_adapters):
            if active_adapter not in lora_A_keys:
                continue
                
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            
            # Extract routing weight for this adapter: [batch, seq, 1]
            weight = routing_weights[..., i:i+1]
            
            x_casted = self._cast_input_dtype(x, lora_A.weight.dtype)
            
            if active_adapter not in self.lora_variant:  # vanilla LoRA
                lora_output = lora_B(lora_A(dropout(x_casted))) * scaling * weight
                result = result + lora_output
            else:
                # For DoRA and other variants, need to adapt
                # For now, fall back to standard application
                lora_output = self.lora_variant[active_adapter].forward(
                    self, active_adapter=active_adapter, x=x_casted, 
                    result=torch.zeros_like(result), **variant_kwargs, **kwargs
                )
                result = result + lora_output * weight
        
        result = result.to(torch_result_dtype)
        return result
    elif adapter_names is not None:
        # ... existing mixed batch logic
```

### B. MoELoraConfig Addition

```python
# In peft/src/peft/tuners/lora/config.py, after LoraConfig

@dataclass
class MoELoraConfig(LoraConfig):
    """
    Configuration class for LoRA with MoE (Mixture-of-Experts) routing support.
    
    Extends LoraConfig with MoE-specific parameters for routing and load balancing.
    """
    
    use_moe_routing: bool = field(
        default=False, 
        metadata={
            "help": "Enable MoE routing for multi-expert inference and training. "
                    "When enabled, routing_weights can be passed to forward() for "
                    "weighted combination of multiple expert adapters."
        }
    )
    
    router_hidden_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Input dimension for router network (typically model's hidden_size). "
                    "Required when use_moe_routing=True."
        }
    )
    
    num_experts: int = field(
        default=1,
        metadata={
            "help": "Number of expert adapters in the MoE. Each expert is a separate "
                    "LoRA adapter sharing the same frozen base model."
        }
    )
    
    top_k_experts: int = field(
        default=1,
        metadata={
            "help": "Number of experts to activate per token (sparse routing). "
                    "Set to num_experts for dense routing. Must be <= num_experts."
        }
    )
    
    router_aux_loss_coef: float = field(
        default=0.01,
        metadata={
            "help": "Coefficient for auxiliary load balancing loss. Helps prevent "
                    "expert collapse where one expert dominates routing."
        }
    )
    
    router_temperature: float = field(
        default=2.0,
        metadata={
            "help": "Temperature for routing softmax. Higher values make routing "
                    "more uniform, lower values make it more peaked."
        }
    )
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.use_moe_routing:
            if self.router_hidden_size is None:
                raise ValueError(
                    "router_hidden_size must be specified when use_moe_routing=True"
                )
            if self.top_k_experts > self.num_experts:
                raise ValueError(
                    f"top_k_experts ({self.top_k_experts}) cannot be greater than "
                    f"num_experts ({self.num_experts})"
                )
            if self.num_experts < 2:
                raise ValueError(
                    f"num_experts must be >= 2 for MoE routing, got {self.num_experts}"
                )
```

## Next Steps

1. Review this plan with stakeholders
2. Set up development branch: `feature/peft-moe-routing`
3. Begin Phase 1 implementation
4. Create PR to PEFT repository after validation
5. Update DyLoRA documentation with migration guide
