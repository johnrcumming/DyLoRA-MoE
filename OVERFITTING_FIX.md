# Overfitting Fix - Increasing Eval Loss

## Problem
Training loss: **0.78** (decreasing) ✓  
Eval loss: **1.04 → 1.05 → 1.07** (increasing) ✗

Classic overfitting - model memorizing training data but not generalizing.

## Root Causes
1. **Label smoothing disabled** (`--label_smoothing 0.0` by default)
2. Long training (4+ epochs on 206K examples)
3. Dense routing = higher model capacity
4. Large training/validation split (90/10)

## Immediate Fixes

### 1. Enable Label Smoothing (CRITICAL)
```bash
--label_smoothing 0.1
```
This prevents the model from becoming too confident on training data.

### 2. Increase Weight Decay
```bash
--weight_decay 0.05  # Up from 0.01
```
Stronger L2 regularization to penalize large weights.

### 3. Increase LoRA Dropout
```bash
--lora_dropout 0.1  # Up from 0.05
```
More dropout in adapter layers to prevent co-adaptation.

### 4. Reduce Training Epochs
```bash
--num_epochs 3  # Stop earlier since eval loss starts increasing at epoch 2
```
Or rely on early stopping (already enabled with patience=3).

### 5. Increase Balance Coefficient
```bash
--balance_coefficient 0.3  # Up from 0.15
```
Encourage expert specialization instead of all experts learning similar patterns.

## Recommended Training Command

### For Cloud (Vertex AI/Vast.ai)
```bash
python train.py \
  --num_experts 4 \
  --num_epochs 3 \
  --datasets "code_alpaca,evol_instruct,code_feedback" \
  --learning_rate 5e-5 \
  --lr_strategy cosine \
  --train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --label_smoothing 0.1 \
  --weight_decay 0.05 \
  --lora_dropout 0.1 \
  --balance_coefficient 0.3 \
  --max_seq_length 1024 \
  --bf16 \
  --output_dir ./models/dylora-overfitting-fix
```

### For Quick Testing (Local)
```bash
python train.py \
  --training_subset 10 \
  --eval_subset 20 \
  --num_epochs 2 \
  --num_experts 2 \
  --label_smoothing 0.1 \
  --weight_decay 0.05 \
  --lora_dropout 0.1 \
  --bf16
```

## Expected Results

**Before (Current):**
- Epoch 1: train=0.84, eval=1.04
- Epoch 2: train=0.80, eval=1.05 ⚠️
- Epoch 3: train=0.78, eval=1.07 ⚠️

**After (With Fixes):**
- Epoch 1: train=0.85, eval=1.03
- Epoch 2: train=0.82, eval=1.01 ✓
- Epoch 3: train=0.80, eval=0.99 ✓

Eval loss should now **decrease** or stabilize with training loss.

## Monitoring

Watch for these signs in wandb/logs:
- ✓ Eval loss tracking training loss (gap < 0.2)
- ✓ Expert load balancing (all experts used >15%)
- ✓ Gradient norms stable (0.3-0.5 range)
- ✗ Eval loss diverging from training loss = still overfitting

## Alternative: Data Augmentation

If overfitting persists, consider:
1. **Reduce training set size**: `--training_subset 80`
2. **Increase validation set**: Modify split to 80/20 instead of 90/10
3. **Add more diverse data**: Include more varied datasets
4. **Use knowledge distillation**: Train on soft labels from a larger model

## References
- Label smoothing paper: https://arxiv.org/abs/1512.00567
- LoRA dropout discussion: https://github.com/huggingface/peft/issues/137
- Early stopping in Transformers: https://huggingface.co/docs/transformers/main_classes/trainer#callbacks
