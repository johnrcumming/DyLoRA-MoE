#!/usr/bin/env python3
"""
DyLoRA-MoE Benchmark Suite

Comprehensive benchmarking system for code generation models using EvalPlus framework.
Supports comparing base models with trained models (local or W&B artifacts).
By default, uses EvalPlus for accurate, standardized evaluation comparable to published benchmarks.

Usage:
    # Basic evaluation with EvalPlus (default)
    python benchmark.py --benchmarks humaneval

    # Compare base model with local trained model
    python benchmark.py --trained_model ./results_full/best_model

    # Compare base model with W&B artifact
    python benchmark.py --wandb_artifact "user/project/artifact:v0"

    # Run specific benchmarks with custom base model
    python benchmark.py --benchmarks humaneval mbpp --model_name microsoft/DialoGPT-medium

    # Run MBPP benchmark only
    python benchmark.py --benchmarks mbpp --max_samples 500

    # Full evaluation (all samples, all benchmarks)
    python benchmark.py --benchmarks humaneval mbpp

    # Quick test (subset, disable W&B logging)
    python benchmark.py --max_samples 10 --no_wandb
    
    # Use legacy custom benchmarks instead of EvalPlus (not recommended, async tests enabled by default)
    python benchmark.py --no_evalplus
"""

import os
import sys
import argparse
import torch
import wandb
import json
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dylo_moe.model import DyLoRA_MoE
from dylo_moe.device_utils import get_device_map, get_torch_dtype, print_device_info
from benchmarks.humaneval_benchmark import HumanEvalBenchmark
from benchmarks.humanevalplus_benchmark import HumanEvalPlusBenchmark
from benchmarks.mbpp_benchmark import MBPPBenchmark
from benchmarks.evalplus_benchmark import EvalPlusBenchmark

# Global dtype override (set by command-line flags)
_dtype_override: Optional[str] = None


def set_dtype_override(dtype: Optional[str]):
    """Set the global dtype override for all model loading."""
    global _dtype_override
    _dtype_override = dtype


def _get_torch_dtype():
    """Get torch dtype with global override applied."""
    return get_torch_dtype(_dtype_override)


def get_base_model_from_config(model_path: str) -> Optional[str]:
    """
    Try to read base_model_name_or_path from config.json in the model directory.
    Returns None if config doesn't exist or doesn't contain the field.
    """
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


def load_base_model(model_name: str, hf_token: Optional[str] = None, force_device: Optional[str] = None):
    """Load base model for comparison."""
    print(f"\n--- Loading Base Model: {model_name} ---")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            device_map=get_device_map(force_device)
        )
        
        print(f"✓ Base model loaded: {model_name}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        return None, None


def load_trained_model(model_path: str, tokenizer=None, hf_token: Optional[str] = None, fallback_base_model: Optional[str] = None, force_device: Optional[str] = None):
    """Load trained DyLoRA-MoE model from local path.
    
    Args:
        model_path: Path to the model directory
        tokenizer: Optional existing tokenizer
        hf_token: HuggingFace token
        fallback_base_model: Base model to use if config doesn't specify one
        force_device: Optional device override ('cpu', 'cuda', 'mps', etc.)
    """
    print(f"\n--- Loading Trained Model: {model_path} ---")
    
    try:
        # First, check if this is a DyLoRA-MoE model by looking at config.json or directory structure
        config_path = os.path.join(model_path, "config.json")
        is_dylora_moe = False
        base_model_name = None
        
        # Check for DyLoRA-MoE state directory (strong indicator)
        # First check in model_path itself (new PEFT checkpoint format)
        dylo_moe_state_dir = os.path.join(model_path, "dylo_moe_state")
        if os.path.exists(dylo_moe_state_dir):
            print("Found DyLoRA-MoE state directory - treating as DyLoRA-MoE model")
            is_dylora_moe = True
        else:
            # Fallback: check parent directory (legacy format)
            parent_dir = os.path.dirname(model_path)
            dylo_moe_state_dir = os.path.join(parent_dir, "dylo_moe_state")
            if os.path.exists(dylo_moe_state_dir):
                print("Found legacy DyLoRA-MoE state directory - treating as DyLoRA-MoE model")
                is_dylora_moe = True
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_type = config.get("model_type")
                    base_model_name = config.get("base_model_name_or_path")
                    
                    if model_type == "dylora-moe":
                        is_dylora_moe = True
                        print("Detected DyLoRA-MoE model format")
                    elif model_type and "dylora" in str(model_type).lower():
                        is_dylora_moe = True
                        print(f"Detected DyLoRA variant: {model_type}")
                    elif not model_type or model_type not in ["gemma", "gemma2", "llama", "gpt2", "bert"]:
                        # Invalid or missing model_type - if we found dylo_moe_state, treat as DyLoRA
                        if os.path.exists(dylo_moe_state_dir):
                            print(f"Config has invalid model_type '{model_type}', but dylo_moe_state exists")
                            is_dylora_moe = True
            except Exception as e:
                print(f"Warning: Could not read config.json: {e}")
                # If we found dylo_moe_state directory, still treat as DyLoRA
                if os.path.exists(dylo_moe_state_dir):
                    is_dylora_moe = True
        
        # Handle DyLoRA-MoE models specially
        if is_dylora_moe:
            # Determine effective base model
            effective_base_model = base_model_name or fallback_base_model
            
            if not effective_base_model:
                print("⚠️  Warning: No base model found in config or provided as fallback. Using default: google/codegemma-2b")
                effective_base_model = "google/codegemma-2b"
            
            print(f"Loading DyLoRA-MoE model with base: {effective_base_model}")
            
            # Get tokenizer first if not provided
            if tokenizer is None:
                if os.path.exists(os.path.join(model_path, "tokenizer.json")):
                    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(effective_base_model, token=hf_token)
                tokenizer.pad_token = tokenizer.eos_token
            
            # Check for PEFT checkpoint structure (saved by PeftCheckpointCallback)
            peft_adapters_dir = os.path.join(model_path, "peft_adapters")
            dylo_moe_state_dir = os.path.join(model_path, "dylo_moe_state")
            
            if os.path.exists(peft_adapters_dir) and os.path.exists(dylo_moe_state_dir):
                print("✓ Found PEFT checkpoint structure (peft_adapters/ and dylo_moe_state/)")
                print("  Loading as proper DyLoRA-MoE PEFT model...")
                
                try:
                    # Load DyLoRA-MoE config
                    # Try new format: dylo_moe_state/config.json
                    dylo_config_path = os.path.join(dylo_moe_state_dir, "config.json")
                    if os.path.exists(dylo_config_path):
                        with open(dylo_config_path, 'r') as f:
                            dylo_config = json.load(f)
                        num_experts = dylo_config.get("num_experts", 4)
                        lora_r = dylo_config.get("lora_r", 16)
                        lora_alpha = dylo_config.get("lora_alpha", 32)
                        lora_dropout = dylo_config.get("lora_dropout", 0.05)
                    else:
                        # Legacy format: check main config.json for _dylora_* fields
                        print("  No dylo_moe_state/config.json found, checking main config for _dylora_* fields...")
                        with open(config_path, 'r') as f:
                            main_config = json.load(f)
                        num_experts = main_config.get("_dylora_num_experts") or main_config.get("num_experts", 4)
                        lora_r = main_config.get("_dylora_lora_r") or main_config.get("lora_r", 16)
                        lora_alpha = main_config.get("_dylora_lora_alpha") or main_config.get("lora_alpha", 32)
                        lora_dropout = main_config.get("_dylora_lora_dropout") or main_config.get("lora_dropout", 0.05)
                    
                    print(f"  Configuration: {num_experts} experts, r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
                    
                    # Reconstruct DyLoRA-MoE model from PEFT adapters
                    print(f"  Reconstructing DyLoRA-MoE from PEFT adapters...")
                    model = DyLoRA_MoE(
                        model_name=effective_base_model,
                        num_experts=num_experts,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        token=hf_token,
                        allow_expert_growth=False,
                        balance_coefficient=0.0  # No balancing during inference
                    )
                    
                    # Load PEFT adapters for each expert
                    from peft import set_peft_model_state_dict
                    for expert_id in range(num_experts):
                        # Try new format: expert_N/ directory with adapter_model files
                        expert_dir = os.path.join(peft_adapters_dir, f"expert_{expert_id}")
                        expert_pt_file = os.path.join(peft_adapters_dir, f"expert_{expert_id}.pt")
                        
                        adapter_name = f"expert_{expert_id}"
                        adapter_weights = None
                        
                        if os.path.exists(expert_dir):
                            # New format: directory with adapter_model.safetensors or adapter_model.bin
                            print(f"  Loading expert {expert_id} adapters from {expert_dir}")
                            
                            adapter_weights_file = os.path.join(expert_dir, "adapter_model.safetensors")
                            if not os.path.exists(adapter_weights_file):
                                adapter_weights_file = os.path.join(expert_dir, "adapter_model.bin")
                            
                            if os.path.exists(adapter_weights_file):
                                if adapter_weights_file.endswith('.safetensors'):
                                    from safetensors.torch import load_file
                                    adapter_weights = load_file(adapter_weights_file)
                                else:
                                    adapter_weights = torch.load(adapter_weights_file, map_location="cpu")
                        
                        elif os.path.exists(expert_pt_file):
                            # Legacy format: expert_N.pt file
                            print(f"  Loading expert {expert_id} adapters from {expert_pt_file}")
                            adapter_weights = torch.load(expert_pt_file, map_location="cpu")
                        
                        if adapter_weights is not None:
                            # Set adapter state dict for this expert
                            model.expert_manager.model.set_adapter(adapter_name)
                            set_peft_model_state_dict(model.expert_manager.model, adapter_weights, adapter_name)
                        else:
                            print(f"    ⚠️  Warning: No adapter weights found for expert {expert_id}")
                    
                    # Load router state
                    router_state_file = os.path.join(dylo_moe_state_dir, "router.pt")
                    if os.path.exists(router_state_file):
                        print(f"  Loading router state from {router_state_file}")
                        router_state = torch.load(router_state_file, map_location="cpu")
                        
                        # Handle nested state dict format (gate_state_dict is nested)
                        if "gate_state_dict" in router_state:
                            # Extract gate weights
                            model.router.gate.load_state_dict(router_state["gate_state_dict"])
                            # Load other router attributes
                            if "expert_maturity" in router_state:
                                model.router.expert_maturity = router_state["expert_maturity"]
                            if "top_k" in router_state:
                                model.router.top_k = router_state["top_k"]
                            if "temperature" in router_state:
                                model.router.temperature = router_state["temperature"]
                        else:
                            # Flat state dict format (older checkpoints)
                            model.router.load_state_dict(router_state)
                    
                    # Set all experts as mature for sparse routing during inference
                    for i in range(num_experts):
                        model.router.set_expert_maturity(i, 1)
                    
                    print("✓ Successfully loaded DyLoRA-MoE PEFT checkpoint")
                    print(f"✓ Trained model loaded from: {model_path}")
                    return model, tokenizer
                    
                except Exception as e:
                    print(f"⚠️  Failed to load PEFT checkpoint structure: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to merged model loading...")
                    # Fall through to merged model loading below
            else:
                # No PEFT checkpoint structure found - check for merged model or standard PEFT format
                print("⚠️  No PEFT checkpoint structure found (missing peft_adapters/ or dylo_moe_state/)")
                
                # Check if weights contain PEFT/LoRA structure (might be standard PEFT format)
                model_file = os.path.join(model_path, "model.safetensors")
                pytorch_model = os.path.join(model_path, "pytorch_model.bin")
                has_peft_structure = False
                
                if os.path.exists(model_file):
                    try:
                        from safetensors import safe_open
                        with safe_open(model_file, framework="pt", device="cpu") as f:
                            keys = list(f.keys())
                            has_peft_structure = any("lora_A" in k or "lora_B" in k or "base_layer" in k for k in keys[:10])
                    except:
                        pass
                elif os.path.exists(pytorch_model):
                    try:
                        state_dict = torch.load(pytorch_model, map_location="cpu")
                        has_peft_structure = any("lora_A" in k or "lora_B" in k or "base_layer" in k for k in list(state_dict.keys())[:10])
                    except:
                        pass
                
                # Try loading as PEFT model if structure indicates it
                if has_peft_structure or os.path.exists(os.path.join(model_path, "adapter_config.json")):
                    print("Detected PEFT/LoRA structure in saved weights - loading with PEFT...")
                    try:
                        from peft import PeftModel, PeftConfig
                        
                        # Load base model first
                        print(f"Loading base model: {effective_base_model}")
                        base_model = AutoModelForCausalLM.from_pretrained(
                            effective_base_model,
                            token=hf_token,
                            torch_dtype=_get_torch_dtype(),
                            device_map=get_device_map(force_device),
                            low_cpu_mem_usage=True
                        )
                        
                        # Try loading PEFT config if it exists
                        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                            model = PeftModel.from_pretrained(base_model, model_path)
                        else:
                            # Load PEFT weights manually
                            print("Loading PEFT weights into base model...")
                            if os.path.exists(model_file):
                                from safetensors.torch import load_file
                                state_dict = load_file(model_file)
                            else:
                                state_dict = torch.load(pytorch_model, map_location="cpu")
                            
                            # Load the PEFT weights
                            base_model.load_state_dict(state_dict, strict=False)
                            model = base_model
                        
                        print("✓ Loaded as PEFT model")
                        
                    except Exception as peft_error:
                        print(f"⚠️  PEFT loading failed: {peft_error}")
                        print("Falling back to standard loading...")
                        has_peft_structure = False  # Fall through to standard loading
                
                if not has_peft_structure:
                    # This should be a merged model saved by trainer.save_model()
                    # Fix or create a valid config.json that AutoModel can understand
                    
                    print(f"Preparing valid config.json for model loading...")
                    
                    # Load base model config to get the correct model_type
                    base_config = AutoConfig.from_pretrained(effective_base_model, token=hf_token)
                    base_config_dict = base_config.to_dict()
                    
                    # Read existing config if it exists
                    existing_config = {}
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                existing_config = json.load(f)
                        except:
                            pass
                    
                    # Merge configs: start with base, overlay existing, ensure critical fields
                    merged_config = base_config_dict.copy()
                    merged_config.update(existing_config)
                    
                    # Ensure critical fields are set correctly
                    merged_config["model_type"] = base_config_dict.get("model_type")  # Use base model's type
                    merged_config["base_model_name_or_path"] = effective_base_model
                    merged_config["architectures"] = base_config_dict.get("architectures", merged_config.get("architectures"))
                    
                    # Preserve DyLoRA-specific metadata for reference
                    if existing_config.get("model_type") == "dylora-moe":
                        merged_config["_dylora_original_type"] = "dylora-moe"
                        merged_config["num_experts"] = existing_config.get("num_experts")
                        merged_config["lora_r"] = existing_config.get("lora_r")
                        merged_config["lora_alpha"] = existing_config.get("lora_alpha")
                    
                    # Write the fixed config
                    with open(config_path, 'w') as f:
                        json.dump(merged_config, f, indent=2)
                    
                    print(f"✓ Created valid config.json with model_type: {merged_config['model_type']}")
                    
                    # Now load with the fixed config
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=_get_torch_dtype(),
                            device_map=get_device_map(force_device),
                            low_cpu_mem_usage=True,
                            trust_remote_code=True  # In case there are custom components
                        )
                        print("✓ Loaded DyLoRA-MoE as merged AutoModelForCausalLM")
                        
                    except Exception as load_error:
                        print(f"⚠️  Standard loading failed: {load_error}")
                        print("Attempting alternative loading method...")
                        
                        # Alternative: Load weights into a fresh base model
                        try:
                            import safetensors.torch
                            
                            # Create a fresh base model
                            print(f"Creating fresh base model: {effective_base_model}")
                            model = AutoModelForCausalLM.from_pretrained(
                                effective_base_model,
                                token=hf_token,
                                torch_dtype=_get_torch_dtype(),
                                device_map=get_device_map(force_device),
                                low_cpu_mem_usage=True
                            )
                            
                            # Load the saved weights
                            weights_file = None
                            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                                weights_file = os.path.join(model_path, "model.safetensors")
                            elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                                weights_file = os.path.join(model_path, "pytorch_model.bin")
                            
                            if weights_file:
                                print(f"Loading weights from: {os.path.basename(weights_file)}")
                                if weights_file.endswith(".safetensors"):
                                    state_dict = safetensors.torch.load_file(weights_file)
                                else:
                                    state_dict = torch.load(weights_file, map_location="cpu")
                                
                                # Load state dict into model
                                model.load_state_dict(state_dict, strict=False)
                                print("✓ Loaded weights into base model")
                            else:
                                print("⚠️  No weight files found, using base model weights")
                                
                        except Exception as alt_error:
                            print(f"❌ Alternative loading also failed: {alt_error}")
                            raise
                
        # Try PEFT adapter format
        elif os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("Detected PEFT adapter format")
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=_get_torch_dtype(),
                device_map=get_device_map(force_device)
            )
            
            # Get tokenizer if not provided
            if tokenizer is None:
                # Try to get base model name from adapter config
                config_path = os.path.join(model_path, "adapter_config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path", "google/codegemma-2b")
                
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
                tokenizer.pad_token = tokenizer.eos_token
                
        # Try regular model format 
        elif os.path.exists(os.path.join(model_path, "model.safetensors")) or os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            print("Detected regular model format")
            
            # Get tokenizer first
            if tokenizer is None:
                if os.path.exists(os.path.join(model_path, "tokenizer.json")):
                    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    # Use base model from config or fallback
                    fallback_model = base_model_name or "google/codegemma-2b"
                    tokenizer = AutoTokenizer.from_pretrained(fallback_model, token=hf_token)
                    tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with increased robustness
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=_get_torch_dtype(),
                device_map=get_device_map(force_device),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder="./offload" if get_device_map(force_device) is None else None
            )
            
            print("✓ Loaded as regular AutoModelForCausalLM")
                
        else:
            # Unknown format
            available_files = os.listdir(model_path)
            raise ValueError(f"Model at {model_path} is in unknown format. Available files: {available_files}")
        
        print(f"✓ Trained model loaded from: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load trained model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_wandb_artifact(artifact_path: str, tokenizer=None, hf_token: Optional[str] = None, fallback_base_model: Optional[str] = None, force_device: Optional[str] = None):
    """Load trained model from W&B artifact. Returns (model, tokenizer, base_model_name)
    
    Args:
        artifact_path: W&B artifact path
        tokenizer: Optional existing tokenizer
        hf_token: HuggingFace token
        fallback_base_model: Base model to use if config doesn't specify one
        force_device: Optional device override ('cpu', 'cuda', 'mps', etc.)
    """
    print(f"\n--- Loading Model from W&B Artifact: {artifact_path} ---")
    
    # Use a separate W&B run for artifact downloading to avoid interfering with main benchmark run
    wandb_run = None
    try:
        # Initialize wandb in a separate context (will use WANDB_API_KEY from environment)
        wandb_run = wandb.init(project="dylo-moe-benchmarking", mode="online", reinit=True)
        
        # Download artifact
        artifact = wandb.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        
        print(f"✓ Artifact downloaded to: {artifact_dir}")
        
        # Finish the download run before loading the model
        if wandb_run:
            wandb_run.finish()
            wandb_run = None
        
        # Look for the best_model subdirectory
        model_path = os.path.join(artifact_dir, "best_model")
        if not os.path.exists(model_path):
            model_path = artifact_dir  # Use root if no best_model subdirectory
        
        # Check for config.json to get base model info (optional)
        config_base_model = get_base_model_from_config(model_path)
        
        # Determine which base model to use
        effective_base_model = config_base_model or fallback_base_model
        
        if not effective_base_model:
            print("⚠️  Warning: No base model specified in config or arguments. Using default: google/codegemma-2b")
            effective_base_model = "google/codegemma-2b"
        else:
            print(f"📄 Using base model: {effective_base_model} (from {'config' if config_base_model else 'argument'})")
        
        # Load the model using the same logic as local loading, but pass the effective base model
        model, model_tokenizer = load_trained_model(model_path, tokenizer, hf_token, effective_base_model, force_device)
        
        return model, model_tokenizer, effective_base_model
        
    except Exception as e:
        print(f"❌ Failed to load W&B artifact: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    finally:
        # Ensure the download run is closed if it's still open
        if wandb_run is not None:
            wandb_run.finish()


def run_benchmarks(models: Dict[str, Any], tokenizer, benchmarks: list, 
                  max_samples: Optional[int] = None, log_to_wandb: bool = False,
                  use_async_tests: bool = False, max_concurrent_tests: Optional[int] = None,
                  use_evalplus: bool = False, evalplus_backend: str = "hf",
                  model_name_for_evalplus: Optional[str] = None,
                  peft_moe_artifact: Optional[str] = None,
                  peft_moe_base_model: Optional[str] = None,
                  peft_moe_routing: str = "router") -> Dict[str, Any]:
    """Run all specified benchmarks on all models.
    
    Args:
        models: Dict of model_name -> model
        tokenizer: HuggingFace tokenizer
        benchmarks: List of benchmark names to run
        max_samples: Max samples per benchmark
        log_to_wandb: Whether to log to W&B
        use_async_tests: Enable async test execution (ignored if use_evalplus=True)
        max_concurrent_tests: Max concurrent tests (ignored if use_evalplus=True)
        use_evalplus: Use native EvalPlus framework
        evalplus_backend: EvalPlus backend (hf, vllm, etc.)
        model_name_for_evalplus: Model name/path for EvalPlus (required if use_evalplus=True)
    """
    
    if use_evalplus:
        # Use native EvalPlus benchmarks
        if not model_name_for_evalplus:
            raise ValueError("model_name_for_evalplus is required when use_evalplus=True")
        
        # Map benchmark names to EvalPlus datasets
        evalplus_dataset_map = {
            'humaneval': 'humaneval',
            'humanevalplus': 'humaneval',  # EvalPlus uses 'humaneval' for both
            'mbpp': 'mbpp'
        }
        
        all_results = {}
        
        for model_name, model in models.items():
            # For peft_moe backend, model is None since EvalPlus loads it directly
            if model is None and evalplus_backend != "peft_moe":
                print(f"⚠️  Skipping {model_name} (failed to load)")
                continue
            
            model_results = {}
            
            for benchmark_name in benchmarks:
                if benchmark_name not in evalplus_dataset_map:
                    raise ValueError(f"Unknown benchmark for EvalPlus: {benchmark_name}")
                
                evalplus_dataset = evalplus_dataset_map[benchmark_name]
                
                # Construct backend-specific kwargs for peft_moe
                backend_kwargs = {}
                if evalplus_backend == "peft_moe":
                    if peft_moe_artifact:
                        # Determine if this is a W&B artifact or local path
                        # W&B artifacts have format: "entity/project/artifact:version"
                        is_wandb_artifact = ":" in peft_moe_artifact and "/" in peft_moe_artifact and not os.path.exists(peft_moe_artifact)
                        
                        if is_wandb_artifact:
                            backend_kwargs["wandb_artifact"] = peft_moe_artifact
                        else:
                            # Local path - use adapter_path instead
                            backend_kwargs["adapter_path"] = peft_moe_artifact
                    if peft_moe_base_model:
                        backend_kwargs["base_model"] = peft_moe_base_model
                    if peft_moe_routing:
                        backend_kwargs["routing_strategy"] = peft_moe_routing
                
                # Create EvalPlus benchmark instance
                benchmark = EvalPlusBenchmark(
                    tokenizer=tokenizer,
                    model_name=model_name_for_evalplus,
                    dataset=evalplus_dataset,
                    max_new_tokens=4096,
                    greedy=True,
                    backend=evalplus_backend,
                    force_base_prompt=False,
                    mini=False,
                    backend_kwargs=backend_kwargs
                )
                
                # Run benchmark (model is passed but not directly used by EvalPlus)
                result = benchmark.run_benchmark(
                    model=model,
                    max_samples=max_samples,
                    log_to_wandb=log_to_wandb,
                    prefix=model_name
                )
                
                model_results[benchmark_name] = result
            
            all_results[model_name] = model_results
        
        return all_results
    
    else:
        # Use legacy custom benchmarks
        available_benchmarks = {
            'humaneval': HumanEvalBenchmark(tokenizer, max_new_tokens=4096, use_adaptive_tokens=False,
                                           use_async_tests=use_async_tests, max_concurrent_tests=max_concurrent_tests),
            'humanevalplus': HumanEvalPlusBenchmark(tokenizer, max_new_tokens=4096, use_adaptive_tokens=False,
                                                   use_async_tests=use_async_tests, max_concurrent_tests=max_concurrent_tests),
            'mbpp': MBPPBenchmark(tokenizer, max_new_tokens=4096, use_adaptive_tokens=False,
                                 use_async_tests=use_async_tests, max_concurrent_tests=max_concurrent_tests)
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
        
        # Print token limit information
        print("\nToken Generation Limits:")
        print("-" * 80)
        for model_name, metrics in model_metrics.items():
            min_limit = metrics.get('min_token_limit', 0)
            avg_limit = metrics.get('avg_token_limit', 0)
            max_limit = metrics.get('max_token_limit', 0)
            avg_generated = metrics.get('avg_tokens_generated', 0)
            print(f"{model_name}:")
            print(f"  Max token limits: min={min_limit:.0f}, avg={avg_limit:.0f}, max={max_limit:.0f}")
            print(f"  Avg tokens generated: {avg_generated:.1f}")
        
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
               "  python benchmark.py --model_name microsoft/phi-2 --trained_model ./results/best_model\n"
               "  python benchmark.py --wandb_artifact user/project/model:v0 --no_wandb  # Disable W&B logging\n"
               "  python benchmark.py --max_samples 20 --bf16  # Force bfloat16 precision\n"
               "  python benchmark.py --max_samples 20 --fp16  # Force float16 precision\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/codegemma-2b",
                       help="Base model name (e.g., 'google/codegemma-2b'). Optional if using --wandb_artifact or --trained_model.")
    parser.add_argument("--trained_model", type=str, default=None,
                       help="Path to trained model directory (e.g., './results_full/best_model')")
    parser.add_argument("--wandb_artifact", type=str, default=None,
                       help="W&B artifact path (e.g., 'user/project/artifact:v0'). Contains model and tokenizer.")
    
    # Benchmark arguments
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["humaneval"],
                       help="Benchmarks to run: humaneval, humanevalplus, mbpp (default: humaneval)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per benchmark (default: all - HumanEval=164, HumanEval+=164, MBPP=500)")
    parser.add_argument("--use_evalplus", action="store_true", default=True,
                       help="Use native EvalPlus framework for generation and evaluation (default: enabled)")
    parser.add_argument("--no_evalplus", action="store_false", dest="use_evalplus",
                       help="Disable EvalPlus and use legacy custom benchmarks")
    parser.add_argument("--evalplus_backend", type=str, default="hf",
                       help="EvalPlus backend: hf (HuggingFace), vllm, peft_moe, etc. (default: hf)")
    
    # PEFT MoE arguments
    parser.add_argument("--peft_moe_artifact", type=str, default=None,
                       help="W&B artifact path for PEFT MoE model (e.g., 'user/project/artifact:v0'). Enables peft_moe backend automatically.")
    parser.add_argument("--peft_moe_base_model", type=str, default=None,
                       help="Base model for PEFT MoE (e.g., 'google/codegemma-2b'). Auto-detected from checkpoint if not provided.")
    parser.add_argument("--peft_moe_routing", type=str, default="router",
                       help="Routing strategy for PEFT MoE: router, single:N, ensemble, best, round_robin (default: router)")
    
    parser.add_argument("--use_async_tests", action="store_true", default=True,
                       help="Enable async test execution for improved GPU utilization (default: enabled for legacy benchmarks with --no_evalplus)")
    parser.add_argument("--no_async_tests", action="store_false", dest="use_async_tests",
                       help="Disable async test execution (only affects legacy benchmarks with --no_evalplus)")
    parser.add_argument("--max_concurrent_tests", type=int, default=None,
                       help="Max concurrent test processes for async mode (default: cpu_count()//2, only for legacy benchmarks with --no_evalplus)")
    
    # Environment arguments
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace token (default: from HF_TOKEN env var)")
    parser.add_argument("--wandb_key", type=str, default=None,
                       help="W&B API key (default: from WANDB_API_KEY env var)")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable W&B logging (default: enabled)")
    
    # Precision arguments
    parser.add_argument("--fp16", action="store_true",
                       help="Force float16 precision (default: auto-detect based on device)")
    parser.add_argument("--bf16", action="store_true",
                       help="Force bfloat16 precision (default: auto-detect based on device)")
    
    # Device argument
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for model inference: 'cpu', 'cuda', 'cuda:0', 'mps', etc. (default: auto-detect)")
    
    return parser.parse_args(argv)


def main(args=None):
    """Main benchmark function. Can be called with pre-parsed args or will parse from command line."""
    if args is None:
        args = parse_args()
    
    # Handle precision flags
    if args.fp16 and args.bf16:
        print("❌ Error: Cannot specify both --fp16 and --bf16")
        sys.exit(1)
    
    dtype_override = None
    if args.fp16:
        dtype_override = 'fp16'
        set_dtype_override('fp16')
        print("ℹ️  Forcing float16 precision")
    elif args.bf16:
        dtype_override = 'bf16'
        set_dtype_override('bf16')
        print("ℹ️  Forcing bfloat16 precision")
    
    # Print device info at startup (with dtype override applied)
    print_device_info(dtype_override)
    
    # Handle peft_moe backend configuration
    if args.peft_moe_artifact:
        # Automatically enable peft_moe backend when artifact is specified
        args.evalplus_backend = "peft_moe"
        args.use_evalplus = True
        print(f"ℹ️  Auto-enabling peft_moe backend for artifact: {args.peft_moe_artifact}")
        print(f"ℹ️  Routing strategy: {args.peft_moe_routing}")
        # For peft_moe, we don't load base model separately - the decoder handles it
        # Skip base model loading
        args.model_name = None
    
    # Validate arguments
    if not args.wandb_artifact and not args.trained_model and not args.peft_moe_artifact:
        print(f"ℹ️  Using default base model: {args.model_name}")
        if not args.model_name:
            print("❌ Error: Must specify at least one of --model_name, --wandb_artifact, --trained_model, or --peft_moe_artifact")
            sys.exit(1)
    
    # Set logging preference (default: log to wandb unless --no_wandb specified)
    log_to_wandb = not args.no_wandb
    
    # Set up environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    
    # Try to determine base model from config files if not explicitly provided
    config_base_model = None
    is_peft_checkpoint = False
    if args.trained_model and os.path.exists(args.trained_model):
        config_base_model = get_base_model_from_config(args.trained_model)
        # Check if this is a PEFT checkpoint (has peft_adapters directory)
        is_peft_checkpoint = os.path.exists(os.path.join(args.trained_model, "peft_adapters"))
    
    # For W&B artifacts, we'll get the config after downloading
    # So we handle that separately below
    
    # Use config base model if found and no explicit model_name argument was provided
    # Check if model_name is the default value
    if config_base_model and args.model_name == "google/codegemma-2b":
        print(f"ℹ️  Using base model from trained model config: {config_base_model}")
        args.model_name = config_base_model
    
    # Auto-enable peft_moe backend for PEFT checkpoints when using EvalPlus
    if is_peft_checkpoint and args.use_evalplus and args.evalplus_backend == "hf":
        print(f"ℹ️  Detected PEFT checkpoint, switching to peft_moe backend")
        args.evalplus_backend = "peft_moe"
        # Store trained_model path for peft_moe backend
        args.peft_moe_artifact = args.trained_model
        args.peft_moe_base_model = config_base_model
        # Don't load model separately - peft_moe decoder handles it
        args.model_name = None
    
    # Prepare models dictionary
    models = {}
    
    # Skip model loading if using EvalPlus with standard backends (hf, vllm, etc.)
    # EvalPlus loads models internally, so pre-loading wastes memory
    skip_model_loading = args.use_evalplus and args.evalplus_backend not in ["peft_moe"]
    
    if skip_model_loading:
        print(f"\nℹ️  Skipping model pre-loading (EvalPlus backend '{args.evalplus_backend}' loads models internally)")
        # Still need to get the base model name for EvalPlus
        if args.trained_model and os.path.exists(args.trained_model):
            config_base_model = get_base_model_from_config(args.trained_model)
            if config_base_model and args.model_name == "google/codegemma-2b":
                print(f"   Using base model from config: {config_base_model}")
                args.model_name = config_base_model
        
        # Load tokenizer only (lightweight)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Loaded tokenizer from: {args.model_name}")
        
        # Create dummy model dict for the benchmark loop
        # EvalPlus will load the actual model
        models = {"evalplus": None}
    
    else:
        # Legacy behavior: Load models for custom benchmarks or peft_moe backend
        # Load base model if specified
        if args.model_name:
            base_model, tokenizer = load_base_model(args.model_name, hf_token, args.device)
            if base_model is None:
                print("❌ Cannot proceed without base model")
                sys.exit(1)
            models["base"] = base_model
        else:
            # We'll get tokenizer from the first loaded model
            tokenizer = None
        
        # Load trained model if specified
        if args.trained_model:
            trained_model, trained_tokenizer = load_trained_model(args.trained_model, tokenizer, hf_token, force_device=args.device)
            if trained_model is not None:
                models["trained"] = trained_model
                if tokenizer is None:
                    tokenizer = trained_tokenizer
        
        # Load W&B artifact if specified
        if args.wandb_artifact:
            wandb_model, wandb_tokenizer, wandb_config_base_model = load_wandb_artifact(
                args.wandb_artifact, tokenizer, hf_token, args.model_name, args.device
            )
            if wandb_model is not None:
                models["wandb"] = wandb_model
                if tokenizer is None:
                    tokenizer = wandb_tokenizer
                
                # If we found a config base model from wandb artifact and haven't loaded base model yet
                if wandb_config_base_model and "base" not in models:
                    print(f"ℹ️  Loading base model from W&B artifact config: {wandb_config_base_model}")
                    base_model, base_tokenizer = load_base_model(wandb_config_base_model, hf_token, args.device)
                    if base_model is not None:
                        models["base"] = base_model
                        if tokenizer is None:
                            tokenizer = base_tokenizer
            else:
                print("⚠️  Warning: W&B artifact model failed to load, but will continue with other models")
                # Even if wandb artifact failed, we might still have base model loaded
                # So don't exit here, just log the warning
    
    # Validate that we have at least one model and tokenizer
    # Exception: peft_moe backend doesn't need pre-loaded models
    if len(models) == 0 and not args.peft_moe_artifact:
        print("❌ No models loaded successfully")
        sys.exit(1)
    
    if tokenizer is None and not args.peft_moe_artifact:
        print("❌ No tokenizer available")
        sys.exit(1)
    
    if args.peft_moe_artifact:
        # For peft_moe, create a dummy models dict for the benchmark loop
        # The actual model will be loaded by EvalPlus
        models = {"peft_moe": None}
        tokenizer = None  # EvalPlus will load tokenizer
        print(f"\n✅ PEFT MoE backend enabled: {args.peft_moe_artifact}")
    else:
        print(f"\n✅ Successfully loaded {len(models)} model(s): {list(models.keys())}")
    
    # Initialize W&B if logging requested (default: True, unless --no_wandb)
    if log_to_wandb:
        wandb.init(project="dylo-moe-benchmarks", 
                  config={
                      "model_name": args.model_name,
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
        if args.use_evalplus:
            print(f"Using EvalPlus framework (backend: {args.evalplus_backend})")
        elif args.use_async_tests:
            print(f"Async test execution: enabled (max workers: {args.max_concurrent_tests or 'auto'})")
        print(f"{'='*100}")
        
        # Determine model name for EvalPlus
        model_name_for_evalplus = None
        if args.use_evalplus:
            # Use the actual model path/name for EvalPlus
            if args.peft_moe_artifact:
                # For peft_moe, use placeholder - decoder will handle model loading
                model_name_for_evalplus = "peft_moe_model"
            elif args.trained_model:
                # For trained models, we need the base model name since EvalPlus loads from HF
                # Try to get it from config, otherwise use args.model_name
                model_name_for_evalplus = get_base_model_from_config(args.trained_model) or args.model_name
            elif args.wandb_artifact:
                # Similar for W&B artifacts
                model_name_for_evalplus = args.model_name
            else:
                # For base model, use the model_name directly
                model_name_for_evalplus = args.model_name
            
            if not args.peft_moe_artifact:
                print(f"EvalPlus will use model: {model_name_for_evalplus}")
        
        results = run_benchmarks(
            models=models,
            tokenizer=tokenizer,
            benchmarks=args.benchmarks,
            max_samples=args.max_samples,
            log_to_wandb=log_to_wandb,
            use_async_tests=args.use_async_tests,
            max_concurrent_tests=args.max_concurrent_tests,
            use_evalplus=args.use_evalplus,
            evalplus_backend=args.evalplus_backend,
            model_name_for_evalplus=model_name_for_evalplus,
            peft_moe_artifact=args.peft_moe_artifact,
            peft_moe_base_model=args.peft_moe_base_model,
            peft_moe_routing=args.peft_moe_routing
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
    # Protect imports for Windows multiprocessing
    # This prevents re-import issues when EvalPlus spawns worker processes
    import multiprocessing
    multiprocessing.freeze_support()
    
    main()