#!/bin/bash
# Validation script for overfitting fixes
# This script runs a quick training to verify that eval loss now tracks training loss
#
# Expected behavior:
#   BEFORE fixes: eval loss increases (1.04 → 1.05 → 1.07) while train loss decreases
#   AFTER fixes: eval loss decreases or stays close to train loss (gap < 0.2)
#
# Usage:
#   ./validate_overfitting_fix.sh

set -e  # Exit on error

echo "=========================================="
echo "DyLoRA-MoE Overfitting Fix Validation"
echo "=========================================="
echo ""
echo "This script validates that the new default hyperparameters fix the overfitting issue."
echo ""
echo "New defaults:"
echo "  - label_smoothing: 0.1 (was 0.0)"
echo "  - weight_decay: 0.05 (was 0.01)"
echo "  - lora_dropout: 0.1 (was 0.05)"
echo ""
echo "Expected results:"
echo "  ✓ Eval loss should track training loss (gap < 0.2)"
echo "  ✓ Eval loss should decrease or stabilize over epochs"
echo "  ✗ Eval loss diverging = still overfitting"
echo ""
echo "=========================================="
echo ""

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable not set"
    echo "   Please run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  Warning: WANDB_API_KEY not set. Training will run without W&B logging."
    echo "   To enable W&B: export WANDB_API_KEY=your_wandb_key"
fi

echo "🚀 Starting validation training..."
echo ""
echo "Configuration:"
echo "  - Data: 10% training, 20% eval (fast validation)"
echo "  - Epochs: 2 (enough to see convergence)"
echo "  - Experts: 2 (minimal MoE setup)"
echo "  - Mixed precision: BF16"
echo ""
echo "This should take ~10-20 minutes on a single GPU."
echo ""
echo "=========================================="
echo ""

# Run training with validation parameters
python train.py \
  --training_subset 10 \
  --eval_subset 20 \
  --num_epochs 2 \
  --num_experts 2 \
  --bf16 \
  --output_dir ./validation_results

echo ""
echo "=========================================="
echo "✅ Validation training complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check the training logs above"
echo "  2. Verify eval loss decreased or stayed close to training loss"
echo "  3. If using W&B, check the run for detailed metrics"
echo ""
echo "Success criteria:"
echo "  ✓ Final eval loss < 1.1 (should be ~0.9-1.0)"
echo "  ✓ Eval loss trend: decreasing or stable"
echo "  ✓ Gap between train/eval loss: < 0.2"
echo ""
echo "If validation passed, you can run full training with:"
echo "  python train.py --bf16 --num_epochs 3 --datasets 'code_alpaca,evol_instruct,code_feedback'"
echo ""
