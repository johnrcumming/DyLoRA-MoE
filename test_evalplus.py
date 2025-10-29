#!/usr/bin/env python3
"""
Quick test of EvalPlus integration (now the default).
Tests a small subset with the base model.
"""
import sys
import os

# Run benchmark with EvalPlus (default) on a small subset
print("Testing EvalPlus integration with google/codegemma-2b...")
print("EvalPlus is now the default - no --use_evalplus flag needed!")
print("This will run HumanEval on first 5 samples only\n")

os.system("""
python benchmark.py \
    --benchmarks humaneval \
    --max_samples 5 \
    --no_wandb
""")
