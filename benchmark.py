#!/usr/bin/env python3
"""
DyLoRA-MoE Benchmark Suite

Comprehensive benchmarking system for evaluating code generation models.
Supports comparing base models with trained models (local or W&B artifacts).
By default, results are logged to Weights & Biases (use --no_wandb to disable).

Usage:
    # Compare base model with local trained model (uses google/codegemma-2b by default)
    python benchmark.py --trained_model ./results_full/best_model

    # Compare base model with W&B artifact (uses google/codegemma-2b by default)
    python benchmark.py --wandb_artifact "user/project/artifact:v0"

    # Run only specific benchmarks with custom base model
    python benchmark.py --benchmarks humaneval --base_model microsoft/DialoGPT-medium

    # Full evaluation (all samples, uses default base model)
    python benchmark.py --max_samples 164

    # Quick test (subset, disable W&B logging, uses default base model)
    python benchmark.py --max_samples 20 --no_wandb
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
from benchmarks.humaneval_benchmark import HumanEvalBenchmark


def get_base_model_from_config(model_path: str) -> Optional[str]:
    """
    Try to read base_model_name_or_path from config.json in the model directory.
    Returns None if config doesn't exist or doesn't contain the field.
    """
    import json
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path")
                if base_model:
                    print(f"📄 Found base model in config: {base_model}")
                    return base_model
        except Exception as e:
            print(f"⚠️  Warning: Could not read config.json: {e}")
    
    return None


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


def load_trained_model(model_path: str, tokenizer=None, hf_token: Optional[str] = None):
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
            
            # Get tokenizer if not provided
            if tokenizer is None:
                # Try to get base model name from adapter config
                import json
                config_path = os.path.join(model_path, "adapter_config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path", "google/codegemma-2b")
                
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
                tokenizer.pad_token = tokenizer.eos_token
                
        elif os.path.exists(os.path.join(model_path, "model.safetensors")) or os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            # Check if this is a DyLoRA-MoE model (has dylo_moe_state directory alongside)
            parent_dir = os.path.dirname(model_path)
            dylo_moe_state_dir = os.path.join(parent_dir, "dylo_moe_state")
            
            if os.path.exists(dylo_moe_state_dir):
                # This is a DyLoRA-MoE model saved from W&B artifact
                print("Detected DyLoRA-MoE model format")
                
                # Get tokenizer first
                if tokenizer is None:
                    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
                        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        # Fallback to default base model
                        tokenizer = AutoTokenizer.from_pretrained("google/codegemma-2b", token=hf_token)
                        tokenizer.pad_token = tokenizer.eos_token
                
                # Load as regular AutoModelForCausalLM (DyLoRA-MoE was saved as full merged model)
                # Use more aggressive memory management for large models
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    low_cpu_mem_usage=True,
                    offload_folder="./offload" if not torch.cuda.is_available() else None
                )
                
                print("✓ Loaded DyLoRA-MoE model as merged AutoModelForCausalLM")
                
            else:
                # Regular full model format (saved with trainer.save_model())
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # Get tokenizer if not provided
                if tokenizer is None:
                    # Try to load tokenizer from the same directory
                    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
                        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        # Fallback to default base model
                        tokenizer = AutoTokenizer.from_pretrained("google/codegemma-2b", token=hf_token)
                        tokenizer.pad_token = tokenizer.eos_token
                
        else:
            # Unknown format
            available_files = os.listdir(model_path)
            raise ValueError(f"Model at {model_path} is in unknown format. Available files: {available_files}")
        
        
        print(f"✓ Trained model loaded from: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load trained model: {e}")
        return None, None


def load_wandb_artifact(artifact_path: str, tokenizer=None, hf_token: Optional[str] = None):
    """Load trained model from W&B artifact. Returns (model, tokenizer, base_model_name)"""
    print(f"\n--- Loading Model from W&B Artifact: {artifact_path} ---")
    
    try:
        # Initialize wandb (will use WANDB_API_KEY from environment)
        wandb.init(project="dylo-moe-benchmarking", mode="online")
        
        # Download artifact
        artifact = wandb.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        
        print(f"✓ Artifact downloaded to: {artifact_dir}")
        
        # Look for the best_model subdirectory
        model_path = os.path.join(artifact_dir, "best_model")
        if not os.path.exists(model_path):
            model_path = artifact_dir  # Use root if no best_model subdirectory
        
        # Check for config.json to get base model info
        config_base_model = get_base_model_from_config(model_path)
        
        # Load the model using the same logic as local loading
        model, model_tokenizer = load_trained_model(model_path, tokenizer, hf_token)
        
        wandb.finish()
        return model, model_tokenizer, config_base_model
        
    except Exception as e:
        print(f"❌ Failed to load W&B artifact: {e}")
        wandb.finish()
        return None, None, None


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


def parse_args(argv=None):
    """Parse command line arguments. Can be called with custom argv for programmatic use."""
    parser = argparse.ArgumentParser(
        description="DyLoRA-MoE Benchmark Suite",
        epilog="Examples:\n"
               "  python benchmark.py --max_samples 20  # Uses default google/codegemma-2b\n"
               "  python benchmark.py --wandb_artifact user/project/model:v0 --max_samples 164\n"
               "  python benchmark.py --trained_model ./results/best_model --max_samples 50\n"
               "  python benchmark.py --wandb_artifact user/project/model:v0 --no_wandb  # Disable W&B logging\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="google/codegemma-2b",
                       help="Base model name (e.g., 'google/codegemma-2b'). Optional if using --wandb_artifact or --trained_model.")
    parser.add_argument("--trained_model", type=str, default=None,
                       help="Path to trained model directory (e.g., './results_full/best_model')")
    parser.add_argument("--wandb_artifact", type=str, default=None,
                       help="W&B artifact path (e.g., 'user/project/artifact:v0'). Contains model and tokenizer.")
    
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
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable W&B logging (default: enabled)")
    
    return parser.parse_args(argv)


def main(args=None):
    """Main benchmark function. Can be called with pre-parsed args or will parse from command line."""
    if args is None:
        args = parse_args()
    
    # Validate arguments
    if not args.wandb_artifact and not args.trained_model:
        print(f"ℹ️  Using default base model: {args.base_model}")
        if not args.base_model:
            print("❌ Error: Must specify at least one of --base_model, --wandb_artifact, or --trained_model")
            sys.exit(1)
    
    # Set logging preference (default: log to wandb unless --no_wandb specified)
    log_to_wandb = not args.no_wandb
    
    # Set up environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    
    # Try to determine base model from config files if not explicitly provided
    config_base_model = None
    if args.trained_model and os.path.exists(args.trained_model):
        config_base_model = get_base_model_from_config(args.trained_model)
    
    # For W&B artifacts, we'll get the config after downloading
    # So we handle that separately below
    
    # Use config base model if found and no explicit base_model argument was provided
    # Check if base_model is the default value
    if config_base_model and args.base_model == "google/codegemma-2b":
        print(f"ℹ️  Using base model from trained model config: {config_base_model}")
        args.base_model = config_base_model
    
    # Prepare models dictionary
    models = {}
    
    # Load base model if specified
    if args.base_model:
        base_model, tokenizer = load_base_model(args.base_model, hf_token)
        if base_model is None:
            print("❌ Cannot proceed without base model")
            sys.exit(1)
        models["base"] = base_model
    else:
        # We'll get tokenizer from the first loaded model
        tokenizer = None
    
    # Load trained model if specified
    if args.trained_model:
        trained_model, trained_tokenizer = load_trained_model(args.trained_model, tokenizer, hf_token)
        if trained_model is not None:
            models["trained"] = trained_model
            if tokenizer is None:
                tokenizer = trained_tokenizer
    
    # Load W&B artifact if specified
    if args.wandb_artifact:
        wandb_model, wandb_tokenizer, wandb_config_base_model = load_wandb_artifact(args.wandb_artifact, tokenizer, hf_token)
        if wandb_model is not None:
            models["wandb"] = wandb_model
            if tokenizer is None:
                tokenizer = wandb_tokenizer
            
            # If we found a config base model from wandb artifact and haven't loaded base model yet
            if wandb_config_base_model and "base" not in models:
                print(f"ℹ️  Loading base model from W&B artifact config: {wandb_config_base_model}")
                base_model, base_tokenizer = load_base_model(wandb_config_base_model, hf_token)
                if base_model is not None:
                    models["base"] = base_model
                    if tokenizer is None:
                        tokenizer = base_tokenizer
    
    # Validate that we have at least one model and tokenizer
    if len(models) == 0:
        print("❌ No models loaded successfully")
        sys.exit(1)
    
    if tokenizer is None:
        print("❌ No tokenizer available")
        sys.exit(1)
    
    # Initialize W&B if logging requested (default: True, unless --no_wandb)
    if log_to_wandb:
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
            log_to_wandb=log_to_wandb
        )
        
        # Compare results
        compare_results(results, args.benchmarks)
        
        # Save results summary
        if len(results) > 0:
            print(f"\n{'='*100}")
            print("BENCHMARK COMPLETED SUCCESSFULLY")
            print(f"{'='*100}")
        
    finally:
        if log_to_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()