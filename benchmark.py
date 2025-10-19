#!/usr/bin/env python3
"""
DyLoRA-MoE Benchmark Suite

Comprehensive benchmarking system for evaluating code generation models.
Supports comparing base models with trained models (local or W&B artifacts).

Usage:
    # Compare base model with local trained model
    python benchmark.py --base_model google/codegemma-2b --trained_model ./results_full/best_model

    # Compare base model with W&B artifact
    python benchmark.py --base_model google/codegemma-2b --wandb_artifact "user/project/artifact:v0"

    # Run only specific benchmarks
    python benchmark.py --benchmarks humaneval --base_model google/codegemma-2b

    # Full evaluation (all samples)
    python benchmark.py --base_model google/codegemma-2b --max_samples 164

    # Quick test (subset)
    python benchmark.py --base_model google/codegemma-2b --max_samples 20
"""

import os
import sys
import argparse
import torch
import wandb
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dylo_moe.model import DyLoRA_MoE
from benchmark.humaneval_benchmark import HumanEvalBenchmark


def load_base_model(model_name: str, hf_token: Optional[str] = None):
    """Load base model for comparison."""
    print(f"\n--- Loading Base Model: {model_name} ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print(f"✓ Base model loaded: {model_name}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return None, None


def load_trained_model(model_path: str, tokenizer, hf_token: Optional[str] = None):
    """Load trained DyLoRA-MoE model from local path."""
    print(f"\n--- Loading Trained Model: {model_path} ---")
    
    try:
        # Load the trained DyLoRA-MoE model
        # This assumes the model was saved with trainer.save_model()
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # PEFT model format
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            # Try to load as DyLoRA_MoE directly
            # This would require reconstruction - for now, use PEFT format
            raise ValueError(f"Model at {model_path} is not in expected PEFT format")
        
        print(f"✓ Trained model loaded from: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load trained model: {e}")
        return None


def load_wandb_artifact(artifact_path: str, tokenizer, hf_token: Optional[str] = None):
    """Load trained model from W&B artifact."""
    print(f"\n--- Loading Model from W&B Artifact: {artifact_path} ---")
    
    try:
        # Initialize wandb (will use WANDB_API_KEY from environment)
        wandb.init(project="dylo-moe-benchmarking", mode="offline")
        
        # Download artifact
        artifact = wandb.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        
        print(f"✓ Artifact downloaded to: {artifact_dir}")
        
        # Look for the best_model subdirectory
        model_path = os.path.join(artifact_dir, "best_model")
        if not os.path.exists(model_path):
            model_path = artifact_dir  # Use root if no best_model subdirectory
        
        # Load the model using the same logic as local loading
        model = load_trained_model(model_path, tokenizer, hf_token)
        
        wandb.finish()
        return model
        
    except Exception as e:
        print(f"❌ Failed to load W&B artifact: {e}")
        wandb.finish()
        return None


def run_benchmarks(models: Dict[str, Any], tokenizer, benchmarks: list, 
                  max_samples: Optional[int] = None, log_to_wandb: bool = False) -> Dict[str, Any]:
    """Run all specified benchmarks on all models."""
    
    # Initialize available benchmarks
    available_benchmarks = {
        'humaneval': HumanEvalBenchmark(tokenizer, max_new_tokens=256)
    }
    
    # Validate requested benchmarks
    for benchmark_name in benchmarks:
        if benchmark_name not in available_benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(available_benchmarks.keys())}")
    
    # Run benchmarks
    all_results = {}
    
    for model_name, model in models.items():
        if model is None:
            print(f"⚠️  Skipping {model_name} (failed to load)")
            continue
            
        model_results = {}
        
        for benchmark_name in benchmarks:
            benchmark = available_benchmarks[benchmark_name]
            
            # Run benchmark
            result = benchmark.run_benchmark(
                model=model,
                max_samples=max_samples,
                log_to_wandb=log_to_wandb,
                prefix=model_name
            )
            
            model_results[benchmark_name] = result
        
        all_results[model_name] = model_results
    
    return all_results


def compare_results(results: Dict[str, Any], benchmarks: list):
    """Print comparison table of results."""
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON")
    print("="*100)
    
    for benchmark_name in benchmarks:
        print(f"\n{benchmark_name.upper()} Results:")
        print("-" * 80)
        
        # Collect metrics for comparison
        model_metrics = {}
        for model_name in results:
            if benchmark_name in results[model_name]:
                metrics = results[model_name][benchmark_name]['metrics']
                model_metrics[model_name] = metrics
        
        if not model_metrics:
            print("No results available for comparison")
            continue
        
        # Print comparison table
        # Get all metric names
        all_metrics = set()
        for metrics in model_metrics.values():
            all_metrics.update(metrics.keys())
        
        # Filter to important numeric metrics
        important_metrics = [m for m in all_metrics 
                           if any(keyword in m.lower() for keyword in 
                                 ['pass@1', 'score', 'rate', 'passed', 'accuracy'])
                           and isinstance(list(model_metrics.values())[0].get(m), (int, float))]
        
        if not important_metrics:
            print("No comparable metrics found")
            continue
        
        # Print header
        print(f"{'Metric':<25}", end="")
        for model_name in model_metrics.keys():
            print(f"{model_name:>15}", end="")
        print()
        print("-" * (25 + 15 * len(model_metrics)))
        
        # Print metrics
        for metric in sorted(important_metrics):
            print(f"{metric:<25}", end="")
            for model_name in model_metrics.keys():
                value = model_metrics[model_name].get(metric, 0)
                if isinstance(value, float):
                    print(f"{value:>15.4f}", end="")
                else:
                    print(f"{value:>15}", end="")
            print()
        
        # Calculate improvements
        if len(model_metrics) == 2:
            model_names = list(model_metrics.keys())
            base_name, trained_name = model_names[0], model_names[1]
            
            print(f"\nImprovement ({trained_name} vs {base_name}):")
            print("-" * 50)
            
            for metric in sorted(important_metrics):
                base_val = model_metrics[base_name].get(metric, 0)
                trained_val = model_metrics[trained_name].get(metric, 0)
                
                if base_val > 0:
                    improvement = ((trained_val - base_val) / base_val) * 100
                    abs_improvement = trained_val - base_val
                    print(f"{metric:<25} {improvement:>+7.1f}% ({abs_improvement:>+.4f})")
                else:
                    print(f"{metric:<25} {'N/A':>15}")


def main():
    parser = argparse.ArgumentParser(description="DyLoRA-MoE Benchmark Suite")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model name (e.g., 'google/codegemma-2b')")
    parser.add_argument("--trained_model", type=str, default=None,
                       help="Path to trained model directory (e.g., './results_full/best_model')")
    parser.add_argument("--wandb_artifact", type=str, default=None,
                       help="W&B artifact path (e.g., 'user/project/artifact:v0')")
    
    # Benchmark arguments
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["humaneval"],
                       help="Benchmarks to run (default: humaneval)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per benchmark (default: all)")
    
    # Environment arguments
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace token (default: from HF_TOKEN env var)")
    parser.add_argument("--wandb_key", type=str, default=None,
                       help="W&B API key (default: from WANDB_API_KEY env var)")
    parser.add_argument("--log_to_wandb", action="store_true",
                       help="Log results to W&B")
    
    args = parser.parse_args()
    
    # Set up environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    
    # Load base model
    base_model, tokenizer = load_base_model(args.base_model, hf_token)
    if base_model is None:
        print("❌ Cannot proceed without base model")
        sys.exit(1)
    
    # Prepare models dictionary
    models = {"base": base_model}
    
    # Load trained model if specified
    if args.trained_model:
        trained_model = load_trained_model(args.trained_model, tokenizer, hf_token)
        if trained_model is not None:
            models["trained"] = trained_model
    
    # Load W&B artifact if specified
    if args.wandb_artifact:
        wandb_model = load_wandb_artifact(args.wandb_artifact, tokenizer, hf_token)
        if wandb_model is not None:
            models["wandb"] = wandb_model
    
    # Validate that we have at least one model
    if len(models) == 0:
        print("❌ No models loaded successfully")
        sys.exit(1)
    
    # Initialize W&B if logging requested
    if args.log_to_wandb:
        wandb.init(project="dylo-moe-benchmarks", 
                  config={
                      "base_model": args.base_model,
                      "trained_model": args.trained_model,
                      "wandb_artifact": args.wandb_artifact,
                      "benchmarks": args.benchmarks,
                      "max_samples": args.max_samples,
                  })
    
    try:
        # Run benchmarks
        print(f"\n{'='*100}")
        print("STARTING BENCHMARK EVALUATION")
        print(f"Models: {list(models.keys())}")
        print(f"Benchmarks: {args.benchmarks}")
        print(f"Max samples: {args.max_samples or 'all'}")
        print(f"{'='*100}")
        
        results = run_benchmarks(
            models=models,
            tokenizer=tokenizer,
            benchmarks=args.benchmarks,
            max_samples=args.max_samples,
            log_to_wandb=args.log_to_wandb
        )
        
        # Compare results
        compare_results(results, args.benchmarks)
        
        # Save results summary
        if len(results) > 0:
            print(f"\n{'='*100}")
            print("BENCHMARK COMPLETED SUCCESSFULLY")
            print(f"{'='*100}")
        
    finally:
        if args.log_to_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()