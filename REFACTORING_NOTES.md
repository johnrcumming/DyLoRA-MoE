# Training Refactoring: From Continual Learning to Joint Training

## Overview
Refactored the DyLoRA-MoE training approach from continual learning (sequential skill-based training) to traditional joint training (all data together).

## Key Changes

### 1. **Dataset Handling**
- **Before**: Datasets were treated as separate "skills" in a sequential data stream
- **After**: Code Alpaca and MBPP datasets are merged into a single combined training set
- **Benefit**: The model sees diverse examples throughout training, allowing the router and experts to learn together

### 2. **Expert Initialization**
- **Before**: Started with 1 expert, dynamically added new experts when "novel" skills were detected
- **After**: All experts (default: 4) are created upfront at initialization
- **Change**: All experts are marked as "mature" from the start, enabling sparse delegation routing from the beginning
- **Benefit**: Consistent architecture throughout training; no mid-training structural changes

### 3. **Training Loop**
- **Before**: Continual learning loop that:
  - Forced creation of new experts for each skill
  - Trained one expert at a time on skill-specific data
  - Required novelty detection and skill library management
- **After**: Single standard training run:
  - One `trainer.train()` call on the combined dataset
  - All experts and the router learn jointly
  - Router learns which expert(s) to use for different types of code naturally
- **Benefit**: Simpler, more stable training; router can learn optimal routing patterns

### 4. **Router Behavior**
- **Before**: Used "dense collaboration" mode when new experts were immature (maturity=0)
- **After**: All experts start mature (maturity=1), so router uses sparse top-k delegation from the start
- **Benefit**: More efficient routing; clearer specialization signals

### 5. **Evaluation Strategy**
- **Maintained**: Separate evaluation on Code Alpaca and MBPP datasets
- **Purpose**: Track per-domain performance to understand if experts specialize naturally
- **Metrics Logged**: 
  - Initial and final losses for both domains
  - Per-epoch evaluation on Code Alpaca (primary eval dataset)

### 6. **Parameter Verification**
- **Added**: Explicit check to ensure all LoRA parameters are trainable
- **Output**: Prints breakdown of trainable vs frozen parameters
- **Safety**: Raises an error if no LoRA parameters are trainable

## Configuration Changes

### Command Line Arguments
- `--num_experts`: Default changed from `1` to `4` (number of experts created at initialization)
- `--allow_expert_growth`: **Removed** (no longer relevant; growth is disabled)

### Model Initialization
- `allow_expert_growth=False` is now hardcoded in `train.py`
- All experts are created upfront and marked mature

## Expected Behavior

### Training
- Loss should decrease steadily as all experts learn simultaneously
- Router weights update to specialize experts on different code patterns
- No structural changes during training (stable architecture)

### Routing
- Router uses sparse top-k delegation (default k=1)
- Router learns to route similar code patterns to the same expert(s)
- Natural specialization emerges (e.g., one expert for Python, another for algorithms)

### Evaluation
- Track both Code Alpaca and MBPP performance separately
- Can observe if certain experts specialize on certain domains

## Migration Guide

### Running the New Training
```bash
# Default: 4 experts, joint training on combined dataset
python train.py --bf16 --num_epochs 10

# Customize number of experts
python train.py --bf16 --num_experts 8 --num_epochs 10

# Use a subset for faster testing
python train.py --bf16 --training_subset 10 --eval_subset 20
```

### What to Monitor
1. **Training loss**: Should decrease steadily (not jump around)
2. **Per-domain eval losses**: Both Code Alpaca and MBPP should improve
3. **Router entropy**: High entropy = all experts used; Low entropy = sparse specialization
4. **Expert utilization**: Check if some experts are underutilized (future work)

## Technical Details

### Multi-Expert Forward Pass
The model still uses the multi-expert routing when `num_experts > 1`:
1. Get transformer outputs for routing decisions
2. Router computes weights for each expert based on hidden states
3. Each expert processes the input independently
4. Outputs are combined using routing weights (averaged across sequence length)

### Why This Should Work Better
1. **Stable gradients**: All parameters train together from the start
2. **Better router learning**: Router sees diverse data and can learn meaningful patterns
3. **Expert specialization**: Emerges naturally from routing decisions, not forced
4. **Simpler debugging**: No complex continual learning dynamics

## Files Modified
- `train.py`: Main training script (major refactor)
- No changes to `dylo_moe/` modules (architecture unchanged)

## Backward Compatibility
- Old continual learning approach is still in the model code (`add_new_skill` method)
- Can be re-enabled by setting `allow_expert_growth=True` and restructuring the training loop
- Not recommended unless specifically needed for continual learning research

## Next Steps
1. Run training and monitor loss curves
2. Analyze expert specialization patterns
3. Consider adding expert utilization metrics to wandb
4. Experiment with different numbers of experts (2, 4, 8)
5. Try different router configurations (top_k, temperature)
