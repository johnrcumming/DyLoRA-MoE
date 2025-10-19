#!/usr/bin/env python3
"""
DyLoRA-MoE Container Entrypoint

Unified entry point for containerized training and benchmarking jobs.
Supports both Google Vertex AI and Vast.ai deployments.

Usage:
  python entrypoint.py --train [training args...]
  python entrypoint.py --benchmark [benchmark args...]

Examples:
  # Training
  python entrypoint.py --train --datasets "code_alpaca,mbpp" --num_epochs 10 --bf16
  
  # Benchmarking
  python entrypoint.py --benchmark --model-path "./results/best_model" --max-samples 164
  
  # W&B artifact benchmarking
  python entrypoint.py --benchmark --wandb-artifact "user/project/model:v0" --max-samples 50
"""

import argparse
import sys
import os
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check that required environment variables are set."""
    required_vars = ["HF_TOKEN"]
    optional_vars = ["WANDB_API_KEY"]
    
    missing_required = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_required.append(var)
    
    if missing_required:
        logger.error(f"Missing required environment variables: {missing_required}")
        logger.error("Please set these variables before running the container:")
        for var in missing_required:
            logger.error(f"  -e {var}=<your_value>")
        sys.exit(1)
    
    # Check optional variables
    for var in optional_vars:
        if not os.environ.get(var):
            logger.warning(f"Optional environment variable {var} not set")
            if var == "WANDB_API_KEY":
                logger.warning("  Weights & Biases logging will be disabled")
    
    logger.info("✓ Environment variables validated")

def detect_platform():
    """Detect if running on Google Vertex AI or Vast.ai."""
    # Google Vertex AI detection
    if os.path.exists("/var/log/gcplogs-docker-driver"):
        return "vertex-ai"
    
    # Check for Google Cloud metadata server
    try:
        import requests
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/",
            headers={"Metadata-Flavor": "Google"},
            timeout=2
        )
        if response.status_code == 200:
            return "vertex-ai"
    except:
        pass
    
    # Vast.ai detection (check for vast-specific environment variables)
    if os.environ.get("VAST_INSTANCE_ID") or os.environ.get("VAST_CONTAINERLABEL"):
        return "vast-ai"
    
    # Check for common vast.ai patterns
    hostname = os.environ.get("HOSTNAME", "")
    if "vast" in hostname.lower():
        return "vast-ai"
    
    # Default to unknown
    return "unknown"

def setup_platform_specific():
    """Setup platform-specific configurations."""
    platform = detect_platform()
    logger.info(f"Detected platform: {platform}")
    
    if platform == "vertex-ai":
        logger.info("Setting up for Google Vertex AI...")
        # Vertex AI specific setup
        # - Usually has good networking and storage
        # - Can use larger batch sizes
        # - Preemptible instances need checkpointing
        os.environ.setdefault("PLATFORM", "vertex-ai")
        
    elif platform == "vast-ai":
        logger.info("Setting up for Vast.ai...")
        # Vast.ai specific setup
        # - Variable hardware quality
        # - May need conservative batch sizes
        # - Storage is often ephemeral
        os.environ.setdefault("PLATFORM", "vast-ai")
        
    else:
        logger.info("Platform unknown, using default settings...")
        os.environ.setdefault("PLATFORM", "unknown")
    
    return platform

def run_training(args):
    """Execute training with provided arguments."""
    logger.info("🚀 Starting DyLoRA-MoE Training")
    
    # Build command for train.py
    cmd = ["python", "train.py"] + args
    
    logger.info(f"Training command: {' '.join(cmd)}")
    
    try:
        # Run training with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            logger.info("✅ Training completed successfully")
        else:
            logger.error(f"❌ Training failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        process.terminate()
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        sys.exit(1)

def run_benchmark(args):
    """Execute benchmarking with provided arguments."""
    logger.info("📊 Starting DyLoRA-MoE Benchmarking")
    
    # Build command for benchmark.py
    cmd = ["python", "benchmark.py"] + args
    
    logger.info(f"Benchmark command: {' '.join(cmd)}")
    
    try:
        # Run benchmark with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            logger.info("✅ Benchmarking completed successfully")
        else:
            logger.error(f"❌ Benchmarking failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
    except KeyboardInterrupt:
        logger.info("🛑 Benchmarking interrupted by user")
        process.terminate()
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Benchmarking failed with error: {e}")
        sys.exit(1)

def add_platform_defaults(args, mode, platform):
    """Add platform-specific default arguments if not provided."""
    args_dict = {}
    
    # Parse existing args to avoid duplicates
    i = 0
    while i < len(args):
        if args[i].startswith('--'):
            key = args[i]
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                args_dict[key] = args[i + 1]
                i += 2
            else:
                args_dict[key] = True
                i += 1
        else:
            i += 1
    
    defaults = []
    
    if mode == "train":
        # Training defaults based on platform
        if platform == "vertex-ai":
            # Vertex AI has reliable hardware
            if "--train_batch_size" not in args_dict:
                defaults.extend(["--train_batch_size", "4"])
            if "--eval_batch_size" not in args_dict:
                defaults.extend(["--eval_batch_size", "4"])
            if "--gradient_accumulation_steps" not in args_dict:
                defaults.extend(["--gradient_accumulation_steps", "32"])
                
        elif platform == "vast-ai":
            # Vast.ai hardware varies, be conservative
            if "--train_batch_size" not in args_dict:
                defaults.extend(["--train_batch_size", "2"])
            if "--eval_batch_size" not in args_dict:
                defaults.extend(["--eval_batch_size", "2"])
            if "--gradient_accumulation_steps" not in args_dict:
                defaults.extend(["--gradient_accumulation_steps", "64"])
        
        # Common training defaults
        if "--num_epochs" not in args_dict:
            defaults.extend(["--num_epochs", "10"])
        if "--num_experts" not in args_dict:
            defaults.extend(["--num_experts", "2"])
        if "--bf16" not in args_dict and "--fp16" not in args_dict:
            defaults.append("--bf16")
            
    elif mode == "benchmark":
        # Benchmarking defaults
        if "--max-samples" not in args_dict:
            defaults.extend(["--max-samples", "164"])  # Full HumanEval
        if "--temperature" not in args_dict:
            defaults.extend(["--temperature", "0.2"])
        if "--max-tokens" not in args_dict:
            defaults.extend(["--max-tokens", "256"])
    
    return args + defaults

def main():
    """Main entrypoint function."""
    print("=" * 60)
    print("🚀 DyLoRA-MoE Container Entrypoint")
    print("=" * 60)
    
    # Check environment first
    check_environment()
    
    # Setup platform-specific configurations
    platform = setup_platform_specific()
    
    # Parse the main command (--train or --benchmark)
    if len(sys.argv) < 2:
        logger.error("Usage: python entrypoint.py {--train|--benchmark} [args...]")
        sys.exit(1)
    
    mode = sys.argv[1]
    remaining_args = sys.argv[2:]
    
    if mode == "--train":
        logger.info("Mode: Training")
        # Add platform-specific defaults
        final_args = add_platform_defaults(remaining_args, "train", platform)
        run_training(final_args)
        
    elif mode == "--benchmark":
        logger.info("Mode: Benchmarking")
        # Add platform-specific defaults
        final_args = add_platform_defaults(remaining_args, "benchmark", platform)
        run_benchmark(final_args)
        
    else:
        logger.error(f"Unknown mode: {mode}")
        logger.error("Usage: python entrypoint.py {--train|--benchmark} [args...]")
        logger.error("")
        logger.error("Examples:")
        logger.error("  # Training")
        logger.error("  python entrypoint.py --train --datasets 'code_alpaca,mbpp' --num_epochs 5")
        logger.error("  ")
        logger.error("  # Benchmarking local model")
        logger.error("  python entrypoint.py --benchmark --model-path './results/best_model'")
        logger.error("  ")
        logger.error("  # Benchmarking W&B artifact")
        logger.error("  python entrypoint.py --benchmark --wandb-artifact 'user/project/model:v0'")
        sys.exit(1)

if __name__ == "__main__":
    main()