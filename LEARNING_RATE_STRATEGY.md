# Learning Rate Strategy for Short Training Runs

## Problem
When training for 1 epoch vs 10 epochs, the learning rate scheduler behaves very differently:
- **10 epochs**: LR has time to warm up (10% of training) and decay smoothly
- **1 epoch**: Warmup takes 10% of the single epoch, then immediately starts decaying
- **Cosine restarts**: With 2 cycles over 1 epoch, the LR bounces up and down quickly, causing loss instability

## Solution: Adaptive Learning Rate Strategy

### New Command Line Arguments

1. `--lr_strategy {auto,constant,linear,cosine,cosine_restarts}` (default: auto)
   - **auto**: Automatically selects optimal strategy based on number of epochs
   - **constant**: Fixed learning rate (best for 1 epoch)
   - **linear**: Linear decay (good for 2-3 epochs)
   - **cosine**: Cosine decay (good for 4+ epochs)
   - **cosine_restarts**: Cosine with restarts (good for long training)

2. `--scale_lr_for_short_training`
   - Reduces learning rate and warmup ratio for short training runs
   - Helps stabilize training for < 4 epochs

### Auto Strategy Selection

| Epochs | Strategy | Learning Rate | Warmup Ratio | Cosine Cycles |
|--------|----------|---------------|--------------|---------------|
| 1      | constant | 3e-5          | 0.05         | N/A           |
| 2-3    | linear   | 4e-5          | 0.1 (0.05*)  | N/A           |
| 4+     | cosine   | 5e-5          | 0.1          | N/A           |
| 4+ (--cosine_restarts) | cosine_restarts | 5e-5 | 0.1 | 2 |

*With `--scale_lr_for_short_training`

### Special Handling for Cosine Restarts

When using cosine restarts with short training:
- **1 epoch**: 0.5 cycles (half cycle - starts high, ends low)
- **2-3 epochs**: 1 cycle (one complete restart)
- **4+ epochs**: 2 cycles (default)

## Usage Examples

### Option 1: Let the system auto-adapt (Recommended)
```bash
# For 1 epoch - automatically uses constant LR
python train.py --num_epochs 1 --bf16

# For 3 epochs - automatically uses linear decay
python train.py --num_epochs 3 --bf16

# For 10 epochs - automatically uses cosine decay
python train.py --num_epochs 10 --bf16
```

### Option 2: Force specific strategy
```bash
# Force constant LR for any duration
python train.py --num_epochs 5 --lr_strategy constant --bf16

# Force cosine restarts with optimized cycles
python train.py --num_epochs 2 --lr_strategy cosine_restarts --bf16
```

### Option 3: Use scaling for short runs
```bash
# Reduces LR and warmup for better stability
python train.py --num_epochs 1 --scale_lr_for_short_training --bf16
```

## Technical Details

### Constant Strategy (1 epoch)
- **Learning Rate**: 3e-5 (lower for stability)
- **Warmup**: 5% of total steps (very short)
- **Decay**: None - maintains constant rate
- **Best for**: Quick experiments, fine-tuning

### Linear Strategy (2-3 epochs)
- **Learning Rate**: 4e-5 (moderate)
- **Warmup**: 10% of total steps (5% with scaling)
- **Decay**: Linear from peak to 0
- **Best for**: Short training runs, quick adaptation

### Cosine Strategy (4+ epochs)
- **Learning Rate**: 5e-5 (full rate)
- **Warmup**: 10% of total steps
- **Decay**: Smooth cosine curve to 0
- **Best for**: Full training runs, best convergence

### Cosine Restarts (any duration)
- **Cycles**: Adaptive based on training length
- **Benefits**: Escapes local minima, explores solution space
- **Trade-off**: May cause oscillations in short training

## Why This Works

1. **Eliminates LR Bouncing**: Constant LR for 1 epoch prevents scheduler-induced instability
2. **Proper Warmup**: Shorter warmup periods for shorter training
3. **Appropriate Decay**: Linear decay for short runs, cosine for longer runs
4. **Adaptive Cycles**: Restart cycles scaled to training duration
5. **Stability Focus**: Lower learning rates for unstable short training scenarios

## Backward Compatibility

- Default behavior (`--lr_strategy auto`) automatically optimizes for your epoch count
- Existing `--cosine_restarts` flag still works but now with intelligent cycle adaptation
- All other training arguments remain unchanged

## Migration from Old Behavior

| Old Command | New Equivalent | Improvement |
|-------------|----------------|-------------|
| `--num_epochs 1` | `--num_epochs 1` (auto) | Stable constant LR instead of bouncing |
| `--num_epochs 10 --cosine_restarts` | `--num_epochs 10 --cosine_restarts` | Better cycle management |
| Any short run | Add `--scale_lr_for_short_training` | Enhanced stability |