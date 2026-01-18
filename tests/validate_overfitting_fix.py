#!/usr/bin/env python3
"""
Validation script for overfitting fixes
This script runs a quick training to verify that eval loss now tracks training loss

Expected behavior:
    BEFORE fixes: eval loss increases (1.04 → 1.05 → 1.07) while train loss decreases
    AFTER fixes: eval loss decreases or stays close to train loss (gap < 0.2)

Usage:
    python validate_overfitting_fix.py
"""

import os
import sys
import subprocess

def print_header(text, char="="):
    """Print a formatted header"""
    border = char * len(text)
    print(f"\n{border}")
    print(text)
    print(f"{border}\n")

def main():
    print_header("DyLoRA-MoE Overfitting Fix Validation")

    print("This script validates that the new default hyperparameters fix the overfitting issue.\n")
    print("New defaults:")
    print("  - label_smoothing: 0.1 (was 0.0)")
    print("  - weight_decay: 0.05 (was 0.01)")
    print("  - lora_dropout: 0.1 (was 0.05)\n")
    print("Expected results:")
    print("  ✓ Eval loss should track training loss (gap < 0.2)")
    print("  ✓ Eval loss should decrease or stabilize over epochs")
    print("  ✗ Eval loss diverging = still overfitting\n")

    # Check for required environment variables
    if not os.getenv("HF_TOKEN"):
        print("❌ Error: HF_TOKEN environment variable not set")
        print("   Please run: export HF_TOKEN=your_huggingface_token")
        sys.exit(1)

    if not os.getenv("WANDB_API_KEY"):
        print("⚠️  Warning: WANDB_API_KEY not set. Training will run without W&B logging.")
        print("   To enable W&B: export WANDB_API_KEY=your_wandb_key\n")

    print("🚀 Starting validation training...\n")
    print("Configuration:")
    print("  - Data: 10% training, 20% eval (fast validation)")
    print("  - Epochs: 2 (enough to see convergence)")
    print("  - Experts: 2 (minimal MoE setup)")
    print("  - Mixed precision: BF16\n")
    print("This should take ~10-20 minutes on a single GPU.\n")

    # Run training with validation parameters
    cmd = [
        "python", "train.py",
        "--training_subset", "10",
        "--eval_subset", "20",
        "--num_epochs", "2",
        "--num_experts", "2",
        "--bf16",
        "--output_dir", "./validation_results"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(e.returncode)

    print_header("✅ Validation training complete!")

    print("Next steps:")
    print("  1. Check the training logs above")
    print("  2. Verify eval loss decreased or stayed close to training loss")
    print("  3. If using W&B, check the run for detailed metrics\n")
    print("Success criteria:")
    print("  ✓ Final eval loss < 1.1 (should be ~0.9-1.0)")
    print("  ✓ Eval loss trend: decreasing or stable")
    print("  ✓ Gap between train/eval loss: < 0.2\n")
    print("If validation passed, you can run full training with:")
    print("  python train.py --bf16 --num_epochs 3 --datasets 'code_alpaca,evol_instruct,code_feedback'\n")

if __name__ == "__main__":
    main()
