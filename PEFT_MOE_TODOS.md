# PEFT MoE Implementation TODOs

**Status Key:**
- ⬜ Not Started
- 🟦 In Progress
- ✅ Complete
- ⏸️ Blocked
- ❌ Cancelled

## Phase 1: PEFT Core Modifications

### 1.1 LoraLayer.forward() Modifications
**Priority:** Critical | **Estimated Time:** 3-4 hours | **Status:** ✅ COMPLETE

- ✅ **Task 1.1.1:** Extract `routing_weights` from kwargs in `Linear.forward()`
  - File: `peft/src/peft/tuners/lora/layer.py`
  - Line: ~779
  - Added: `routing_weights = kwargs.pop("routing_weights", None)`
  
- ✅ **Task 1.1.2:** Implement MoE routing branch
  - File: `peft/src/peft/tuners/lora/layer.py`
  - Line: ~805 (after adapter_names check)
  - Added conditional: `elif routing_weights is not None:`
  - Implemented weighted adapter combination loop
  
- ✅ **Task 1.1.3:** Handle dtype casting for routing weights
  - Ensured routing_weights dtype matches layer computation dtype
  - Tested with mixed precision (bf16, fp32)
  
- ✅ **Task 1.1.4:** Add support for LoRA variants (DoRA, etc.)
  - Checked if `active_adapter in self.lora_variant`
  - Applied variant-specific forward with weight multiplication
  
- ✅ **Task 1.1.5:** Validate broadcasting semantics
  - routing_weights shape: [batch, seq_len, num_experts]
  - weight slice shape: [batch, seq_len, 1]
  - Verified correct broadcasting with lora output: [batch, seq_len, hidden_dim]

- ✅ **Task 1.1.6:** Implement pre-forward hook propagation
  - File: `peft/src/peft/peft_model.py` - Added "routing_weights" to special_peft_forward_args
  - File: `peft/src/peft/tuners/lora/model.py` - Added _routing_weights_pre_forward_hook
  - File: `peft/src/peft/tuners/lora/model.py` - Modified _enable_peft_forward_hooks to register routing hook
  - Verified routing_weights propagates to all LoRA layers via trace_routing_weights.py

**Acceptance Criteria:**
- [x] Forward pass with routing_weights produces correct output shape
- [x] Backward pass computes gradients for routing_weights
- [x] Fallback to existing logic when routing_weights=None
- [x] routing_weights properly propagates through model layers via hooks
- [x] All tests pass (test_routing_weights.py)

### 1.2 MoELoraConfig Creation
**Priority:** High | **Estimated Time:** 2 hours | **Status:** ✅ COMPLETE

- ✅ **Task 1.2.1:** Create MoELoraConfig dataclass
  - File: `peft/src/peft/tuners/lora/config.py`
  - Location: After LoraConfig class definition (lines 801-1016)
  - Inherits from LoraConfig
  
- ✅ **Task 1.2.2:** Add MoE-specific fields
  - `use_moe_routing: bool = True` (default True for MoE config)
  - `num_experts: int = 4`
  - `top_k_experts: int = 2`
  - `router_hidden_size: Optional[int] = None`
  - `router_aux_loss_coef: float = 0.01`
  - `router_temperature: float = 2.0`
  - `router_type: Literal["learned", "learned_with_maturity", "fixed"] = "learned_with_maturity"`
  - `expert_capacity_factor: float = 1.25`
  - `expert_dropout: float = 0.0`
  - `load_balance_loss_type: Literal["aux_loss", "z_loss", "switch"] = "aux_loss"`
  
- ✅ **Task 1.2.3:** Implement __post_init__ validation
  - Validates num_experts >= 2 when use_moe_routing=True
  - Validates top_k_experts <= num_experts
  - Validates top_k_experts >= 1
  - Validates router_temperature > 0
  - Validates router_aux_loss_coef >= 0
  - Validates expert_capacity_factor > 0
  - Validates expert_dropout in [0, 1)
  - Validates router_type in allowed values
  - Validates load_balance_loss_type in allowed values
  - Warns when MoE params set but use_moe_routing=False
  
- ✅ **Task 1.2.4:** Update config serialization
  - Inherits to_dict() from LoraConfig (tested)
  - All MoE fields included in serialized dict
  
- ✅ **Task 1.2.5:** Add to __init__.py exports
  - File: `peft/src/peft/tuners/lora/__init__.py` - Added import and export
  - File: `peft/src/peft/tuners/__init__.py` - Added import and export
  - File: `peft/src/peft/__init__.py` - Added import and export
  - MoELoraConfig now importable from `peft` top-level

**Acceptance Criteria:**
- [x] MoELoraConfig can be instantiated with valid parameters
- [x] Validation raises appropriate errors for invalid configs (10 tests)
- [x] Config serializes correctly (to_dict tested)
- [x] Config can be imported from peft and peft.tuners.lora
- [x] Comprehensive test suite (test_moe_lora_config.py) with 10 tests all passing

### 1.3 Router Management in LoraModel
**Priority:** High | **Estimated Time:** 4 hours

- ⬜ **Task 1.3.1:** Add router attribute to LoraModel.__init__
  - File: `peft/src/peft/tuners/lora/model.py`
  - Line: ~140 (in __init__)
  - Add: `self.router = None`
  - Add: `self.moe_config = None`
  
- ⬜ **Task 1.3.2:** Implement attach_router() method
  - Location: LoraModel class
  - Signature: `def attach_router(self, router_module: nn.Module, moe_config: Optional[MoELoraConfig] = None)`
  - Store router reference
  - Move router to appropriate device
  
- ⬜ **Task 1.3.3:** Implement detach_router() method
  - Location: LoraModel class
  - Clear router reference
  - Clear moe_config reference
  
- ⬜ **Task 1.3.4:** Implement get_routing_weights() helper
  - Location: LoraModel class
  - Signature: `def get_routing_weights(self, hidden_states: torch.Tensor) -> torch.Tensor`
  - Call self.router(hidden_states)
  - Handle router=None case
  
- ⬜ **Task 1.3.5:** Update state_dict saving to include router
  - Override `state_dict()` to include router weights
  - Key: "router.*" for router parameters
  
- ⬜ **Task 1.3.6:** Update load_state_dict to restore router
  - Override `load_state_dict()` to restore router
  - Handle missing router keys gracefully (backward compat)
  
- ⬜ **Task 1.3.7:** Update device management
  - Ensure router moves with model during .to() calls
  - Handle device_map scenarios

**Acceptance Criteria:**
- [ ] Router can be attached and detached
- [ ] Router persists across save/load cycles
- [ ] Router moves to correct device with model
- [ ] Works with both CPU and GPU
- [ ] Backward compatible with models without routers

## Phase 2: DyLoRA Integration

### 2.1 Update DyLoRA_MoE Forward Pass
**Priority:** Critical | **Estimated Time:** 5 hours

- ⬜ **Task 2.1.1:** Remove expert loop from forward()
  - File: `dylo_moe/model.py`
  - Lines: ~224-245 (for loop through experts)
  - Delete: Expert iteration and weighted accumulation logic
  
- ⬜ **Task 2.1.2:** Implement single-pass routing logic
  - After getting hidden_states from base model
  - Compute routing_weights via self.router(hidden_states)
  - Pass routing_weights to foundation_model() call
  
- ⬜ **Task 2.1.3:** Update active_adapters management
  - Ensure all expert adapters are in active_adapters list
  - Remove set_active_expert() calls in multi-expert branch
  
- ⬜ **Task 2.1.4:** Update monitoring and logging
  - Keep self.last_routing_weights tracking
  - Verify load_balancing_loss computation still works
  - Update any expert-specific metrics
  
- ⬜ **Task 2.1.5:** Handle edge cases
  - Single expert case (no routing needed)
  - Training vs inference mode differences
  - expert_id parameter for single-expert training

**Acceptance Criteria:**
- [ ] Forward pass completes without errors
- [ ] Output logits shape matches expected
- [ ] Loss computation works correctly
- [ ] Training converges to reasonable loss
- [ ] Routing weights are tracked for monitoring

### 2.2 Update ExpertManager
**Priority:** Medium | **Estimated Time:** 2 hours

- ⬜ **Task 2.2.1:** Add activate_all_experts() method
  - File: `dylo_moe/expert.py`
  - Method to set all experts as active simultaneously
  - Return list of expert adapter names
  
- ⬜ **Task 2.2.2:** Update adapter tracking
  - Maintain list of all expert adapter names
  - Ensure PEFT model has correct active_adapters
  
```

### 1.3 Router Management in LoraModel
**Priority:** High | **Estimated Time:** 4 hours

- ⬜ **Task 1.3.1:** Add router attribute to LoraModel.__init__
  - File: `peft/src/peft/tuners/lora/model.py`
  - Line: ~140 (in __init__)
  - Add: `self.router = None`
  - Add: `self.moe_config = None`
  
- ⬜ **Task 1.3.2:** Implement attach_router() method
  - Location: LoraModel class
  - Signature: `def attach_router(self, router_module: nn.Module, moe_config: Optional[MoELoraConfig] = None)`
  - Store router reference
  - Move router to appropriate device
  
- ⬜ **Task 1.3.3:** Implement detach_router() method
  - Location: LoraModel class
  - Clear router reference
  - Clear moe_config reference
  
- ⬜ **Task 1.3.4:** Implement get_routing_weights() helper
  - Location: LoraModel class
  - Signature: `def get_routing_weights(self, hidden_states: torch.Tensor) -> torch.Tensor`
  - Call self.router(hidden_states)
  - Handle router=None case
  
- ⬜ **Task 1.3.5:** Update state_dict saving to include router
  - Override `state_dict()` to include router weights
  - Key: "router.*" for router parameters
  
- ⬜ **Task 1.3.6:** Update load_state_dict to restore router
  - Override `load_state_dict()` to restore router
  - Handle missing router keys gracefully (backward compat)
  
- ⬜ **Task 1.3.7:** Update device management
  - Ensure router moves with model during .to() calls
  - Handle device_map scenarios

**Acceptance Criteria:**
- [ ] Router can be attached and detached
- [ ] Router persists across save/load cycles
- [ ] Router moves to correct device with model
- [ ] Works with both CPU and GPU
- [ ] Backward compatible with models without routers

## Phase 2: DyLoRA Integration

### 2.1 Update DyLoRA_MoE Forward Pass
**Priority:** Critical | **Estimated Time:** 5 hours

- ⬜ **Task 2.1.1:** Remove expert loop from forward()
  - File: `dylo_moe/model.py`
  - Lines: ~224-245 (for loop through experts)
  - Delete: Expert iteration and weighted accumulation logic
  
- ⬜ **Task 2.1.2:** Implement single-pass routing logic
  - After getting hidden_states from base model
  - Compute routing_weights via self.router(hidden_states)
  - Pass routing_weights to foundation_model() call
  
- ⬜ **Task 2.1.3:** Update active_adapters management
  - Ensure all expert adapters are in active_adapters list
  - Remove set_active_expert() calls in multi-expert branch
  
- ⬜ **Task 2.1.4:** Update monitoring and logging
  - Keep self.last_routing_weights tracking
  - Verify load_balancing_loss computation still works
  - Update any expert-specific metrics
  
- ⬜ **Task 2.1.5:** Handle edge cases
  - Single expert case (no routing needed)
  - Training vs inference mode differences
  - expert_id parameter for single-expert training

**Acceptance Criteria:**
- [ ] Forward pass completes without errors
- [ ] Output logits shape matches expected
- [ ] Loss computation works correctly
- [ ] Training converges to reasonable loss
- [ ] Routing weights are tracked for monitoring

### 2.2 Update ExpertManager
**Priority:** Medium | **Estimated Time:** 2 hours

- ⬜ **Task 2.2.1:** Add activate_all_experts() method
  - File: `dylo_moe/expert.py`
  - Method to set all experts as active simultaneously
  - Return list of expert adapter names
  
- ⬜ **Task 2.2.2:** Update adapter tracking
  - Maintain list of all expert adapter names
  - Ensure PEFT model has correct active_adapters
  
- ⬜ **Task 2.2.3:** Update expert creation
  - Verify new experts are added to active list when needed
  - Test with dynamic expert growth

**Acceptance Criteria:**
- [ ] All experts can be activated at once
- [ ] Single expert activation still works
- [ ] Expert creation doesn't break MoE mode

### 2.3 Update Initialization
**Priority:** Low | **Estimated Time:** 1 hour

- ⬜ **Task 2.3.1:** Set all experts as active by default
  - File: `dylo_moe/model.py`
  - In __init__, after creating experts
  - Call expert_manager.activate_all_experts()
  
- ⬜ **Task 2.3.2:** Verify router device placement
  - Ensure router is on same device as model
  - Test with CUDA and CPU

**Acceptance Criteria:**
- [ ] Model initializes with all experts active
- [ ] Router is on correct device
- [ ] No device mismatch errors during forward pass

## Phase 3: Testing & Validation

### 3.1 Unit Tests for PEFT
**Priority:** Critical | **Estimated Time:** 6 hours

- ⬜ **Task 3.1.1:** Create test file
  - File: `peft/tests/test_moe_routing.py`
  - Set up test fixtures and utilities
  
- ⬜ **Task 3.1.2:** Test routing_weights forward pass
  - Test name: `test_routing_weights_forward_basic`
  - Create simple LoRA model with 2 adapters
  - Pass routing_weights, verify output shape
  - Verify output is weighted combination
  
- ⬜ **Task 3.1.3:** Test gradient flow
  - Test name: `test_routing_weights_gradients`
  - Verify gradients computed for routing_weights
  - Verify gradients computed for lora_A, lora_B
  - Check gradient magnitudes are reasonable
  
- ⬜ **Task 3.1.4:** Test backward compatibility
  - Test name: `test_backward_compatibility_no_routing`
  - Call forward without routing_weights
  - Verify existing behavior unchanged
  - Run with all existing PEFT tests
  
- ⬜ **Task 3.1.5:** Test multiple experts
  - Test name: `test_multiple_experts_2_4_8`
  - Test with 2, 4, 8 expert adapters
  - Verify correct weighted combination
  
- ⬜ **Task 3.1.6:** Test routing_weights shapes
  - Test name: `test_routing_weights_various_shapes`
  - Test different batch sizes: 1, 4, 16
  - Test different sequence lengths: 8, 128, 512
  - Verify broadcasting works correctly
  
- ⬜ **Task 3.1.7:** Test mixed precision
  - Test name: `test_routing_weights_mixed_precision`
  - Test with fp16, bf16, fp32
  - Verify no dtype mismatches
  
- ⬜ **Task 3.1.8:** Test router save/load
  - Test name: `test_router_persistence`
  - Attach router, save model
  - Load model, verify router restored
  - Verify router produces same outputs

**Acceptance Criteria:**
- [ ] All tests pass
- [ ] Code coverage >80% for modified code
- [ ] Tests run in CI/CD pipeline

### 3.2 Integration Tests for DyLoRA
**Priority:** Critical | **Estimated Time:** 6 hours

- ⬜ **Task 3.2.1:** Create test file
  - File: `tests/test_dylora_moe_integration.py`
  - Set up fixtures for small model and dataset
  
- ⬜ **Task 3.2.2:** Test output equivalence
  - Test name: `test_single_vs_multi_pass_output_equivalence`
  - Create model with both old and new forward logic
  - Run same inputs through both
  - Compare outputs (should be very close, allowing for numerical precision)
  
- ⬜ **Task 3.2.3:** Test training convergence
  - Test name: `test_training_convergence_small_dataset`
  - Use tiny dataset (100 samples)
  - Train for 10 steps
  - Verify loss decreases
  - Verify routing weights are learnable
  
- ⬜ **Task 3.2.4:** Test memory efficiency
  - Test name: `test_memory_usage_comparison`
  - Measure peak memory with old approach
  - Measure peak memory with new approach
  - Verify new approach uses <10% more memory
  
- ⬜ **Task 3.2.5:** Test speed improvement
  - Test name: `test_forward_pass_speedup`
  - Benchmark old approach (multiple passes)
  - Benchmark new approach (single pass)
  - Verify >1.5x speedup for 4 experts
  
- ⬜ **Task 3.2.6:** Test single expert mode
  - Test name: `test_single_expert_unchanged`
  - Verify single expert training still works
  - Verify no performance regression
  
- ⬜ **Task 3.2.7:** Test expert_id parameter
  - Test name: `test_expert_id_training`
  - Train specific expert via expert_id param
  - Verify only that expert's parameters update
  
- ⬜ **Task 3.2.8:** Test load balancing loss
  - Test name: `test_load_balancing_loss_computation`
  - Verify balance_loss computed correctly
  - Verify it affects routing weights during training

**Acceptance Criteria:**
- [ ] All integration tests pass
- [ ] Performance improvements verified
- [ ] No regressions in existing functionality

### 3.3 Validation Script
**Priority:** High | **Estimated Time:** 4 hours

- ⬜ **Task 3.3.1:** Create validation script
  - File: `validate_moe_modifications.py`
  - CLI interface with argparse
  
- ⬜ **Task 3.3.2:** Implement checkpoint loading
  - Load existing DyLoRA checkpoint
  - Support both old and new checkpoint formats
  
- ⬜ **Task 3.3.3:** Implement comparison logic
  - Run inference with old approach
  - Run inference with new approach
  - Compute output difference metrics
  
- ⬜ **Task 3.3.4:** Implement benchmarking
  - Measure forward pass time (old vs new)
  - Measure memory usage (old vs new)
  - Measure training time per step
  
- ⬜ **Task 3.3.5:** Implement reporting
  - Generate comparison report
  - Include speedup metrics
  - Include memory savings
  - Include output similarity scores
  
- ⬜ **Task 3.3.6:** Add visualization
  - Plot routing weights distribution
  - Plot expert usage over time
  - Plot speedup vs num_experts

**Acceptance Criteria:**
- [ ] Script runs without errors
- [ ] Generates comprehensive validation report
- [ ] Confirms output equivalence
- [ ] Documents performance improvements

## Phase 4: Documentation & Polish

### 4.1 Code Documentation
**Priority:** Medium | **Estimated Time:** 3 hours

- ⬜ **Task 4.1.1:** Add docstrings to modified functions
  - LoraLayer.forward() - document routing_weights parameter
  - LoraModel.attach_router() - document usage
  - MoELoraConfig - document all fields
  
- ⬜ **Task 4.1.2:** Add inline comments
  - Explain MoE routing branch logic
  - Document why specific design choices were made
  
- ⬜ **Task 4.1.3:** Update type hints
  - Add routing_weights type hint to forward signatures
  - Ensure all new code has proper type annotations

**Acceptance Criteria:**
- [ ] All public APIs have docstrings
- [ ] Complex logic has explanatory comments
- [ ] Type hints pass mypy checks

### 4.2 User Documentation
**Priority:** Medium | **Estimated Time:** 4 hours

- ⬜ **Task 4.2.1:** Create MoE routing guide
  - File: `peft/docs/source/moe_routing.md`
  - Explain MoE routing concept
  - Show basic usage example
  - Document configuration options
  
- ⬜ **Task 4.2.2:** Update DyLoRA README
  - File: `README.md`
  - Add section on new routing approach
  - Include performance comparison
  - Link to migration guide
  
- ⬜ **Task 4.2.3:** Create migration guide
  - File: `MIGRATION_GUIDE.md`
  - Step-by-step upgrade instructions
  - Code comparison (before/after)
  - Troubleshooting common issues
  
- ⬜ **Task 4.2.4:** Add example notebook
  - File: `examples/moe_routing_example.ipynb`
  - End-to-end MoE training example
  - Visualization of routing weights
  - Performance benchmarks

**Acceptance Criteria:**
- [ ] Documentation is clear and comprehensive
- [ ] Examples run without errors
- [ ] Migration path is well-documented

### 4.3 Code Quality
**Priority:** Medium | **Estimated Time:** 2 hours

- ⬜ **Task 4.3.1:** Run code formatters
  - Black for Python formatting
  - isort for import sorting
  
- ⬜ **Task 4.3.2:** Run linters
  - flake8 for style checks
  - pylint for code quality
  - Fix any issues found
  
- ⬜ **Task 4.3.3:** Run type checker
  - mypy for type checking
  - Fix type errors
  
- ⬜ **Task 4.3.4:** Check test coverage
  - pytest-cov for coverage report
  - Aim for >80% coverage on modified code

**Acceptance Criteria:**
- [ ] Code passes all quality checks
- [ ] No linter warnings
- [ ] Test coverage meets threshold

## Phase 5: Integration & Deployment

### 5.1 PEFT Pull Request
**Priority:** High | **Estimated Time:** 8 hours

- ⬜ **Task 5.1.1:** Create feature branch
  - Branch name: `feature/moe-routing-support`
  - Base: PEFT main branch
  
- ⬜ **Task 5.1.2:** Commit changes with clear messages
  - Separate commits for each logical change
  - Use conventional commit format
  
- ⬜ **Task 5.1.3:** Write comprehensive PR description
  - Explain motivation and use case
  - Link to design document
  - Include before/after examples
  - Add benchmark results
  
- ⬜ **Task 5.1.4:** Address review feedback
  - Respond to comments
  - Make requested changes
  - Re-test after changes
  
- ⬜ **Task 5.1.5:** Get approval and merge
  - Wait for maintainer review
  - Merge when approved

**Acceptance Criteria:**
- [ ] PR is submitted to PEFT repository
- [ ] All CI checks pass
- [ ] PR is approved by maintainers
- [ ] PR is merged

### 5.2 DyLoRA Update
**Priority:** High | **Estimated Time:** 2 hours

- ⬜ **Task 5.2.1:** Update PEFT dependency
  - Update requirements.txt
  - Pin to version with MoE support
  
- ⬜ **Task 5.2.2:** Update DyLoRA code
  - Implement new forward pass
  - Remove old multi-pass logic
  
- ⬜ **Task 5.2.3:** Update tests
  - Ensure all existing tests pass
  - Add new tests for MoE routing
  
- ⬜ **Task 5.2.4:** Tag new release
  - Version bump (e.g., v2.0.0)
  - Create GitHub release
  - Document breaking changes if any

**Acceptance Criteria:**
- [ ] DyLoRA works with new PEFT version
- [ ] All tests pass
- [ ] Release is published

### 5.3 Performance Validation
**Priority:** Critical | **Estimated Time:** 4 hours

- ⬜ **Task 5.3.1:** Run full training benchmark
  - Train on Code Alpaca dataset
  - Compare old vs new approach
  - Document training time reduction
  
- ⬜ **Task 5.3.2:** Run inference benchmark
  - Test on MBPP evaluation set
  - Compare inference speed
  - Verify output quality unchanged
  
- ⬜ **Task 5.3.3:** Memory profiling
  - Profile memory usage during training
  - Profile memory usage during inference
  - Verify no memory leaks
  
- ⬜ **Task 5.3.4:** Publish benchmark results
  - Update README with benchmarks
  - Create benchmark report document
  - Share on social media / blog

**Acceptance Criteria:**
- [ ] Benchmarks show expected improvements
- [ ] No performance regressions
- [ ] Results are documented and published

## Quick Reference Checklist

### Pre-Implementation
- [ ] Review implementation plan
- [ ] Set up development environment
- [ ] Create feature branches
- [ ] Notify stakeholders of upcoming changes

### During Implementation
- [ ] Follow TODOs in order
- [ ] Write tests alongside code
- [ ] Document as you go
- [ ] Commit frequently with clear messages

### Before PR
- [ ] All tests pass locally
- [ ] Code is formatted and linted
- [ ] Documentation is complete
- [ ] Benchmarks show improvements

### Post-Merge
- [ ] Update DyLoRA dependency
- [ ] Run validation script
- [ ] Update documentation
- [ ] Announce changes to users

## Time Estimates Summary

| Phase | Estimated Time |
|-------|---------------|
| Phase 1: PEFT Core | 9-10 hours |
| Phase 2: DyLoRA Integration | 8 hours |
| Phase 3: Testing | 16 hours |
| Phase 4: Documentation | 9 hours |
| Phase 5: Deployment | 14 hours |
| **Total** | **56-57 hours** (~1.5 weeks full-time) |

## Risk Mitigation

### High Risk Items
1. **Routing weights gradient flow** - Test early, validate with synthetic data
2. **Backward compatibility** - Extensive testing with existing models
3. **Performance regression** - Benchmark at each step

### Contingency Plans
- If gradient flow issues: Add explicit gradient checkpointing
- If compatibility breaks: Add feature flag to toggle new behavior
- If performance worse: Profile and optimize critical paths

## Progress Tracking

Update this section as work progresses:

**Last Updated:** 2025-10-27

**Overall Progress:** 0/90 tasks complete (0%)

**Phase Progress:**
- Phase 1: 0/15 tasks (0%)
- Phase 2: 0/9 tasks (0%)
- Phase 3: 0/21 tasks (0%)
- Phase 4: 0/11 tasks (0%)
- Phase 5: 0/11 tasks (0%)

**Blockers:** None

**Notes:** Implementation plan and TODOs created. Ready to begin Phase 1.
