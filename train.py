import torch
import os
import json
import argparse
os.environ["WANDB_DISABLED"] = "false"
import wandb
from huggingface_hub import login as hf_login
from tqdm import tqdm
import math
import time
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
)
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers.tokenization_utils_base import BatchEncoding
from typing import Union, Dict, Iterable, Any

# DyLoRA-MoE uses PEFT library for training support:
# - ExpertManager wraps model with get_peft_model() and manages multiple adapters
# - Standard Trainer works seamlessly with PEFT models (no custom trainer needed)
# - Checkpoints use PEFT's save_pretrained() for proper adapter serialization
# - All experts share frozen base model weights (handled automatically by PEFT)
from dylo_moe.model import DyLoRA_MoE
from dylo_moe.utils import print_trainable_parameters, save_dylo_moe_state, save_lora_experts
from dylo_moe.device_utils import move_model_to_device, print_device_info, get_device
from benchmarks.humaneval_benchmark import HumanEvalBenchmark
from benchmarks.humanevalplus_benchmark import HumanEvalPlusBenchmark
from benchmarks.mbpp_benchmark import MBPPBenchmark
from benchmarks.evalplus_benchmark import EvalPlusBenchmark
from data.prepare_data import (
    download_mbpp, 
    download_code_alpaca,
    download_humaneval,
    get_dataset,
    AVAILABLE_DATASETS,
)


def check_environment_variables():
    """
    Check for required environment variables and provide helpful error messages.
    Returns the tokens for use in initialization.
    """
    missing_vars = []
    warnings = []
    
    # Check for HF_TOKEN (required for model access)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        missing_vars.append("HF_TOKEN")
    
    # Check for WANDB_API_KEY (optional but recommended)
    wandb_token = os.environ.get("WANDB_API_KEY")
    if not wandb_token:
        warnings.append("WANDB_API_KEY")
    
    # Print errors and warnings
    if missing_vars:
        print("\n❌ ERROR: Required environment variables are missing:")
        for var in missing_vars:
            if var == "HF_TOKEN":
                print(f"   - {var}: Required for accessing Hugging Face models")
                print("     Get your token from: https://huggingface.co/settings/tokens")
                print("     Set it with: export HF_TOKEN=your_token_here")
        print("\nTraining cannot proceed without these variables.")
        exit(1)
    
    if warnings:
        print("\n⚠️  WARNING: Recommended environment variables are missing:")
        for var in warnings:
            if var == "WANDB_API_KEY":
                print(f"   - {var}: Recommended for experiment tracking and logging")
                print("     Get your key from: https://wandb.ai/authorize")
                print("     Set it with: export WANDB_API_KEY=your_key_here")
        print("\nTraining will continue, but some features may be limited.\n")
    else:
        print("✓ All environment variables are properly configured.\n")
    
    return hf_token, wandb_token


def initialize_services(hf_token, wandb_token):
    """
    Initialize Weights & Biases and Hugging Face Hub with provided tokens.
    """
    print("--- Initializing Services ---")
    
    # Initialize Hugging Face Hub
    if hf_token:
        try:
            hf_login(token=hf_token, add_to_git_credential=True)
            print("✓ Hugging Face Hub initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Hugging Face Hub: {e}")
            print("Training may fail when accessing gated models")
    
    # Initialize Weights & Biases
    if wandb_token:
        try:
            wandb.login(key=wandb_token)
            print("✓ Weights & Biases initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize Weights & Biases: {e}")
            print("Experiment tracking may be limited")
    else:
        print("⚠️  Weights & Biases will use existing login or run anonymously")
    
    print()


class GradientMonitoringCallback(TrainerCallback):
    """
    Callback to monitor per-expert gradient norms during training.
    Logs gradient statistics for each expert and the router to wandb.
    """
    
    def __init__(self, model, num_experts: int):
        self.model = model
        self.num_experts = num_experts
        
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Called before optimizer.step() - gradients are still available here."""
        if state.global_step % args.logging_steps == 0:
            # Compute per-expert gradient norms
            expert_grad_norms = {f"expert_{i}": 0.0 for i in range(self.num_experts)}
            router_grad_norm = 0.0
            total_grad_norm = 0.0
            lora_total = 0.0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    grad_norm_sq = param.grad.norm().item() ** 2
                    total_grad_norm += grad_norm_sq
                    
                    # Check if this is a LoRA parameter and which expert it belongs to
                    if "lora" in name.lower():
                        lora_total += grad_norm_sq
                        # Extract expert ID from parameter name
                        # Names look like: ...lora_A.expert_0.weight or ...lora_B.expert_1.weight
                        for i in range(self.num_experts):
                            expert_key = f".expert_{i}"  # Note the dot prefix
                            if expert_key in name:
                                expert_grad_norms[f"expert_{i}"] += grad_norm_sq
                                break
                    
                    # Check if this is a router parameter
                    if "router" in name.lower() or "gate" in name.lower():
                        router_grad_norm += grad_norm_sq
            
            # Compute final norms (square root)
            total_grad_norm = total_grad_norm ** 0.5
            router_grad_norm = router_grad_norm ** 0.5
            lora_total = lora_total ** 0.5
            for expert_key in expert_grad_norms:
                expert_grad_norms[expert_key] = expert_grad_norms[expert_key] ** 0.5
            
            # Log to wandb
            log_dict = {
                "grad_norm/total": total_grad_norm,
                "grad_norm/router": router_grad_norm,
                "grad_norm/lora_all": lora_total,
            }
            for expert_key, norm in expert_grad_norms.items():
                log_dict[f"grad_norm/{expert_key}"] = norm
            
            # Also log gradient norm ratios for analysis
            if total_grad_norm > 0:
                for expert_key, norm in expert_grad_norms.items():
                    if norm > 0:
                        log_dict[f"grad_ratio/{expert_key}"] = norm / total_grad_norm
                if router_grad_norm > 0:
                    log_dict["grad_ratio/router"] = router_grad_norm / total_grad_norm
                if lora_total > 0:
                    log_dict["grad_ratio/lora_all"] = lora_total / total_grad_norm
            
            # Use commit=False to allow other callbacks to add more metrics to the same step
            wandb.log(log_dict, step=state.global_step, commit=False)


class DyLoRAMonitoringCallback(TrainerCallback):
    """
    Callback to monitor DyLoRA-MoE specific metrics during training.
    Tracks routing statistics, expert usage patterns, and routing entropy.
    """
    
    def __init__(self, model, num_experts: int):
        self.model = model
        self.num_experts = num_experts
        # Track cumulative expert usage across training
        self.expert_usage_counts = torch.zeros(num_experts)
        self.total_routing_steps = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step to collect routing statistics."""
        # Only log on logging_steps intervals, but use commit=False to avoid step conflicts
        # The Trainer will commit the logs, so we just add our metrics to the same step
        if state.global_step % args.logging_steps == 0:
            # Get the last routing weights from the model
            # These are stored during forward pass as [batch, seq_len, num_experts]
            if hasattr(self.model, 'last_routing_weights') and self.model.last_routing_weights is not None:
                routing_weights = self.model.last_routing_weights  # [batch, seq_len, num_experts]
                
                # Compute statistics over the batch and sequence
                # Average routing weights across sequence dimension
                batch_avg_weights = routing_weights.mean(dim=1)  # [batch, num_experts]
                
                # 1. Expert Usage Distribution (which experts are being used)
                # Sum weights across batch to get total usage per expert
                expert_usage = batch_avg_weights.sum(dim=0)  # [num_experts]
                total_weight = expert_usage.sum()
                
                if total_weight > 0:
                    expert_usage_normalized = expert_usage / total_weight
                    
                    # Log per-expert usage percentages
                    log_dict = {}
                    for i in range(self.num_experts):
                        log_dict[f"routing/expert_{i}_usage"] = expert_usage_normalized[i].item()
                    
                    # Update cumulative usage counts
                    self.expert_usage_counts += expert_usage.cpu()
                    self.total_routing_steps += 1
                    
                    # 2. Routing Entropy (how diverse is the routing?)
                    # Higher entropy = more balanced expert usage
                    # Lower entropy = routing concentrated on few experts
                    probs = expert_usage_normalized.clamp(min=1e-10)  # Avoid log(0)
                    entropy = -(probs * torch.log(probs)).sum().item()
                    max_entropy = math.log(self.num_experts)  # Maximum possible entropy
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
                    
                    log_dict["routing/entropy"] = entropy
                    log_dict["routing/entropy_normalized"] = normalized_entropy
                    
                    # 3. Load Balancing Metric
                    # Coefficient of variation: std/mean
                    # Lower = more balanced, Higher = more imbalanced
                    std = expert_usage_normalized.std().item()
                    mean = expert_usage_normalized.mean().item()
                    load_balance_cv = std / mean if mean > 0 else 0.0
                    log_dict["routing/load_imbalance"] = load_balance_cv
                    
                    # 4. Expert Dominance (what's the max expert usage?)
                    max_usage = expert_usage_normalized.max().item()
                    log_dict["routing/max_expert_usage"] = max_usage
                    
                    # 5. Active Experts Count (how many experts have >5% usage)
                    active_threshold = 0.05
                    active_experts = (expert_usage_normalized > active_threshold).sum().item()
                    log_dict["routing/active_experts"] = active_experts
                    
                    # 6. Routing Temperature Effect
                    # Check if routing is becoming more or less confident over time
                    # Compute the maximum routing weight per token (across experts)
                    max_weights_per_token = routing_weights.max(dim=-1)[0]  # [batch, seq_len]
                    avg_max_weight = max_weights_per_token.mean().item()
                    log_dict["routing/avg_max_confidence"] = avg_max_weight
                    
                    # 7. Cumulative Expert Usage Statistics
                    if self.total_routing_steps > 0:
                        cumulative_usage = self.expert_usage_counts / self.expert_usage_counts.sum()
                        for i in range(self.num_experts):
                            log_dict[f"routing/cumulative_expert_{i}"] = cumulative_usage[i].item()
                    
                    # 8. Expert Specialization Indicator
                    # Compute variance in routing weights across sequence positions
                    # High variance might indicate experts are specializing for certain positions
                    token_variance = routing_weights.var(dim=1).mean(dim=0)  # [num_experts]
                    for i in range(self.num_experts):
                        log_dict[f"routing/expert_{i}_token_variance"] = token_variance[i].item()
                    
                    # 9. Load Balancing Loss Components (if available)
                    # Log separate LM loss and balance loss for monitoring
                    if hasattr(self.model, 'last_lm_loss') and self.model.last_lm_loss is not None:
                        log_dict["loss/lm_loss"] = self.model.last_lm_loss.item()
                    
                    if hasattr(self.model, 'last_balance_loss') and self.model.last_balance_loss is not None:
                        log_dict["loss/balance_loss"] = self.model.last_balance_loss.item()
                        log_dict["loss/balance_weight"] = self.model.balance_coefficient
                    
                    # 10. Routing Confidence Statistics
                    # Max routing weight indicates confidence in expert selection
                    max_routing_weights = routing_weights.max(dim=-1)[0]  # [batch, seq_len]
                    log_dict["routing/confidence_mean"] = max_routing_weights.mean().item()
                    log_dict["routing/confidence_std"] = max_routing_weights.std().item()
                    log_dict["routing/confidence_min"] = max_routing_weights.min().item()
                    log_dict["routing/confidence_max"] = max_routing_weights.max().item()
                    
                    # Log all metrics to wandb with commit=False
                    # This avoids step number conflicts with Trainer's logging
                    # The Trainer will commit all metrics together
                    wandb.log(log_dict, step=state.global_step, commit=False)
                else:
                    # Edge case: no routing weights (shouldn't happen in normal training)
                    wandb.log({
                        "routing/warning": 1.0,
                        "routing/total_weight": 0.0
                    }, step=state.global_step, commit=False)


class BenchmarkCallback(TrainerCallback):
    """
    Callback to run benchmarks at the end of each epoch.
    Only used when benchmark_strategy='epoch'.
    """
    
    def __init__(self, model, tokenizer, benchmark_names, use_test_execution=True, 
                 use_evalplus=True, evalplus_backend="hf", model_name=None):
        self.model = model
        self.tokenizer = tokenizer
        self.benchmark_names = benchmark_names
        self.use_test_execution = use_test_execution
        self.use_evalplus = use_evalplus
        self.evalplus_backend = evalplus_backend
        self.model_name = model_name
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch to run benchmarks."""
        from dylo_moe.device_utils import get_device
        epoch = int(state.epoch) if state.epoch is not None else 0
        device = get_device()
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} EVALUATION")
        print(f"Benchmarking device: {device.upper()}")
        print(f"{'='*80}")
        
        # Run benchmark suite
        from train import run_benchmark_suite
        epoch_results = run_benchmark_suite(
            model=self.model,
            tokenizer=self.tokenizer,
            benchmark_names=self.benchmark_names,
            max_samples=None,
            use_test_execution=self.use_test_execution,
            log_prefix=f"epoch_{epoch}",
            use_evalplus=self.use_evalplus,
            evalplus_backend=self.evalplus_backend,
            model_name=self.model_name
        )
        
        # Log results to wandb for each benchmark
        for benchmark_name, result in epoch_results.items():
            metrics = result['metrics']
            wandb.log({
                f"epoch_{epoch}/{benchmark_name}_pass@1": metrics.get('pass@1', 0.0),
                f"epoch_{epoch}/{benchmark_name}_syntax_score": metrics.get('syntax_score', 0.0),
                f"epoch_{epoch}/{benchmark_name}_entry_point_score": metrics.get('entry_point_score', 0.0),
                f"epoch_{epoch}/{benchmark_name}_truncation_rate": metrics.get('truncation_rate', 0.0),
                f"epoch_{epoch}/{benchmark_name}_avg_tokens_generated": metrics.get('avg_tokens_generated', 0.0),
                f"epoch_{epoch}/{benchmark_name}_tests_passed": metrics.get('tests_passed', 0),
                f"epoch_{epoch}/{benchmark_name}_tests_run": metrics.get('tests_run', 0),
            }, step=state.global_step, commit=False)
        
        print(f"{'='*80}\n")


class PeftCheckpointCallback(TrainerCallback):
    """
    Callback to save complete MoE-PEFT model using PEFT's conventions.
    
    Saves PEFT adapters using library's native save_pretrained() method and extends
    it to include router state for MoE models. This ensures:
    - LoRA adapters saved in standard PEFT format (adapter_config.json + weights)
    - Router saved alongside adapters for complete MoE checkpoint
    - Compatible with PEFT's attach_router() pattern
    """
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        
    def on_save(self, args, state, control, **kwargs):
        """
        Called when a checkpoint is saved.
        Saves complete MoE-PEFT model (adapters + router + config).
        """
        model = kwargs.get('model')
        if model is None:
            print("⚠️  Warning: Model not found in callback kwargs, skipping checkpoint")
            return
        
        # Determine checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n💾 Saving complete MoE-PEFT checkpoint-{state.global_step}...")
        
        # Get the PEFT model from DyLoRA_MoE wrapper
        peft_model = model.expert_manager.model
        
        # Save PEFT adapters using library's native save_pretrained()
        peft_adapters_dir = os.path.join(checkpoint_dir, "peft_adapters")
        try:
            # Save each expert adapter separately using PEFT's save_pretrained
            for expert_id in range(self.num_experts):
                adapter_name = f"expert_{expert_id}"
                expert_dir = os.path.join(peft_adapters_dir, adapter_name)
                
                # Set this adapter as active before saving
                peft_model.set_adapter(adapter_name)
                
                # Save using PEFT's native method (saves adapter_config.json + adapter_model.safetensors)
                peft_model.save_pretrained(expert_dir, selected_adapters=[adapter_name])
            
            print(f"   ✓ {self.num_experts} PEFT adapters saved")
        except Exception as e:
            print(f"   ✗ Failed to save PEFT adapters: {e}")
            import traceback
            traceback.print_exc()
        
        # Save router state (extends PEFT to support MoE)
        # Following PEFT's naming convention: router.safetensors alongside adapters
        router_path = os.path.join(checkpoint_dir, "router.safetensors")
        try:
            if hasattr(model, 'router'):
                # Save router state dict using safetensors (PEFT's preferred format)
                from safetensors.torch import save_file
                router_state = model.router.state_dict()
                save_file(router_state, router_path)
                print(f"   ✓ Router saved: {router_path}")
            else:
                print(f"   ⚠️  No router found in model")
        except Exception as e:
            print(f"   ✗ Failed to save router: {e}")
            # Fallback to torch.save if safetensors fails
            try:
                torch.save(model.router.state_dict(), router_path.replace('.safetensors', '.pt'))
                print(f"   ✓ Router saved (PyTorch format)")
            except:
                pass
        
        # Create a minimal config.json with DyLoRA-MoE markers
        # This prevents the Trainer from saving a full model config
        config_path = os.path.join(checkpoint_dir, "config.json")
        try:
            # Get base model name from the foundation model
            base_model_name = getattr(model.foundation_model.config, '_name_or_path', 'google/codegemma-2b')
            
            minimal_config = {
                "model_type": "dylora-moe",
                "_dylora_format": "peft_separated",
                "_dylora_checkpoint_step": state.global_step,
                "base_model_name_or_path": base_model_name,
                "num_experts": self.num_experts,
                "checkpoint_type": "peft_only",
                "_note": "This is a PEFT-only checkpoint. Full model weights are NOT saved to save disk space."
            }
            
            with open(config_path, 'w') as f:
                json.dump(minimal_config, f, indent=2)
            
            print(f"   ✓ Created minimal config.json (PEFT-only checkpoint)")
        except Exception as e:
            print(f"   ✗ Failed to create config.json: {e}")
        
        # CRITICAL: Delete merged model files if they exist (Trainer may have saved them already)
        # This prevents the massive 11GB model.safetensors or pytorch_model.bin files
        merged_model_files = [
            os.path.join(checkpoint_dir, "model.safetensors"),
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
            os.path.join(checkpoint_dir, "model.safetensors.index.json"),
            os.path.join(checkpoint_dir, "pytorch_model.bin.index.json"),
        ]
        
        files_deleted = []
        for model_file in merged_model_files:
            if os.path.exists(model_file):
                try:
                    file_size_mb = os.path.getsize(model_file) / (1024 * 1024)
                    os.remove(model_file)
                    files_deleted.append((os.path.basename(model_file), file_size_mb))
                except Exception as e:
                    print(f"   ⚠️  Failed to delete {model_file}: {e}")
        
        if files_deleted:
            total_saved_mb = sum(size for _, size in files_deleted)
            print(f"   🗑️  Deleted merged model files (saved {total_saved_mb:.1f} MB):")
            for filename, size_mb in files_deleted:
                print(f"      - {filename} ({size_mb:.1f} MB)")
        
        print(f"✓ Complete MoE-PEFT checkpoint saved for step {state.global_step}")
        print(f"  Structure: adapters/ + router.safetensors")
        print(f"  Compatible with PEFT's attach_router() pattern\n")
        
        return control
        print(f"   (Checkpoint size: ~100MB adapters + router, NOT 2.6GB merged model)\n")
        
        return control


def analyze_truncation_stats(texts, tokenizer, max_length=768, dataset_name="Dataset"):
    """
    Analyze and report truncation statistics for a dataset.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        dataset_name: Name for reporting
    """
    print(f"\n--- Truncation Analysis: {dataset_name} ---")
    
    token_lengths = []
    truncated_count = 0
    
    for text in texts:
        tokens = tokenizer(text, add_special_tokens=True)['input_ids']
        token_len = len(tokens)
        token_lengths.append(token_len)
        if token_len > max_length:
            truncated_count += 1
    
    truncation_rate = truncated_count / len(texts) if len(texts) > 0 else 0
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_seq_length = max(token_lengths) if token_lengths else 0
    min_seq_length = min(token_lengths) if token_lengths else 0
    
    # Calculate percentiles
    sorted_lengths = sorted(token_lengths)
    p50 = sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0
    p75 = sorted_lengths[int(len(sorted_lengths) * 0.75)] if sorted_lengths else 0
    p90 = sorted_lengths[int(len(sorted_lengths) * 0.90)] if sorted_lengths else 0
    p95 = sorted_lengths[int(len(sorted_lengths) * 0.95)] if sorted_lengths else 0
    p99 = sorted_lengths[int(len(sorted_lengths) * 0.99)] if sorted_lengths else 0
    
    print(f"Total samples: {len(texts)}")
    print(f"Max length setting: {max_length} tokens")
    print(f"")
    print(f"Token length statistics:")
    print(f"  Average: {avg_length:.1f} tokens")
    print(f"  Min: {min_seq_length} tokens")
    print(f"  Max: {max_seq_length} tokens")
    print(f"  Median (p50): {p50} tokens")
    print(f"  p75: {p75} tokens")
    print(f"  p90: {p90} tokens")
    print(f"  p95: {p95} tokens")
    print(f"  p99: {p99} tokens")
    print(f"")
    print(f"Truncation:")
    print(f"  Truncated samples: {truncated_count}/{len(texts)} ({truncation_rate:.1%})")
    
    if truncation_rate > 0.1:
        print(f"  ⚠️  WARNING: {truncation_rate:.1%} of samples will be truncated!")
        print(f"  Consider increasing max_length to {p95} (p95) or {p99} (p99)")
    elif truncation_rate > 0.05:
        print(f"  ⚠️  Note: {truncation_rate:.1%} of samples will be truncated")
    else:
        print(f"  ✓ Low truncation rate: {truncation_rate:.1%}")
    
    print("-" * 60)
    
    return {
        'truncation_rate': truncation_rate,
        'truncated_count': truncated_count,
        'avg_length': avg_length,
        'max_length': max_seq_length,
        'p50': p50,
        'p75': p75,
        'p90': p90,
        'p95': p95,
        'p99': p99,
    }


def preprocess_evaluation_dataset(tokenizer, dataset, max_length: int = 768):
    """
    Tokenizes the evaluation dataset.
    """
    def tokenize_function(examples):
        if "text" in examples:
            # The 'text' column in MBPP is a list of strings, so we join them.
            processed_text = ["\n".join(text) for text in examples["text"]]
        else:
            # The Code Alpaca dataset has 'instruction' and 'output' columns.
            processed_text = [f"{instruction}\n{output}" for instruction, output in zip(examples["instruction"], examples["output"])]
        return tokenizer(processed_text, padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

def pack_sequences(tokenizer, texts, max_length: int = 768):
    """Naive sequence packing: concatenate tokenized sequences until max_length reached."""
    input_ids_batches = []
    attn_batches = []
    cur = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            ids = ids[:max_length]
        if len(cur) + len(ids) > max_length:
            if cur:
                pad_len = max_length - len(cur)
                input_ids_batches.append(cur + [tokenizer.pad_token_id] * pad_len)
                attn_batches.append([1] * len(cur) + [0] * pad_len)
            cur = []
        cur.extend(ids)
    if cur:
        pad_len = max_length - len(cur)
        input_ids_batches.append(cur + [tokenizer.pad_token_id] * pad_len)
        attn_batches.append([1] * len(cur) + [0] * pad_len)
    return input_ids_batches, attn_batches

def create_interleaved_dataset(dataset1, dataset2, total_samples=None):
    """
    Create an interleaved dataset with equal representation from both datasets.
    
    This fixes dataset imbalance by ensuring 50/50 sampling probability regardless
    of original dataset sizes. Helps prevent bias and enables better expert specialization.
    
    Args:
        dataset1: First dataset (e.g., Code Alpaca examples)
        dataset2: Second dataset (e.g., MBPP examples)
        total_samples: Total number of samples to generate (default: len(dataset1) + len(dataset2))
    
    Returns:
        List of interleaved examples with balanced representation
    """
    import random
    
    if total_samples is None:
        total_samples = len(dataset1) + len(dataset2)
    
    interleaved = []
    
    # Use separate indices for each dataset to allow with-replacement sampling
    # This ensures we can always provide 50/50 even if one dataset is much smaller
    dataset1_indices = list(range(len(dataset1)))
    dataset2_indices = list(range(len(dataset2)))
    
    for _ in range(total_samples):
        # 50/50 probability of selecting from each dataset
        if random.random() < 0.5:
            # Sample from dataset1 (with replacement if needed)
            idx = random.choice(dataset1_indices)
            interleaved.append(dataset1[idx])
        else:
            # Sample from dataset2 (with replacement if needed)
            idx = random.choice(dataset2_indices)
            interleaved.append(dataset2[idx])
    
    # Log sampling statistics
    dataset1_count = sum(1 for ex in interleaved if ex in dataset1)
    dataset2_count = sum(1 for ex in interleaved if ex in dataset2)
    
    print(f"\n--- Interleaved Dataset Statistics ---")
    print(f"Dataset 1 samples: {dataset1_count} ({dataset1_count/total_samples*100:.1f}%)")
    print(f"Dataset 2 samples: {dataset2_count} ({dataset2_count/total_samples*100:.1f}%)")
    print(f"Total samples: {total_samples}")
    print(f"Balance ratio: {min(dataset1_count, dataset2_count) / max(dataset1_count, dataset2_count):.2f}")
    
    return interleaved


def create_multi_dataset_interleaved(*datasets, total_samples=None):
    """
    Create an interleaved dataset from multiple datasets with equal representation.
    
    Args:
        *datasets: Variable number of datasets to interleave
        total_samples: Total number of samples to generate (default: sum of all dataset sizes)
    
    Returns:
        List of tuples (dataset_index, item) with equal representation from each dataset
    """
    import random
    
    num_datasets = len(datasets)
    if num_datasets == 0:
        return []
    
    if total_samples is None:
        total_samples = sum(len(d) for d in datasets)
    
    # Equal probability for each dataset
    dataset_prob = 1.0 / num_datasets
    
    # Create interleaved dataset
    interleaved = []
    dataset_indices = [0] * num_datasets  # Track position in each dataset
    
    for _ in range(total_samples):
        # Choose dataset with equal probability
        dataset_idx = random.choices(range(num_datasets), weights=[dataset_prob] * num_datasets)[0]
        
        # Sample from chosen dataset (with replacement if needed)
        item_idx = dataset_indices[dataset_idx] % len(datasets[dataset_idx])
        item = datasets[dataset_idx][item_idx]
        
        interleaved.append((dataset_idx, item))
        dataset_indices[dataset_idx] += 1
    
    # Log statistics
    counts = [sum(1 for idx, _ in interleaved if idx == i) for i in range(num_datasets)]
    print(f"\nMulti-dataset interleaving statistics:")
    print(f"Total samples: {len(interleaved)}")
    for i, count in enumerate(counts):
        print(f"  Dataset {i+1}: {count} samples ({count/len(interleaved)*100:.1f}%)")
    
    # Calculate balance (should be close to 1.0 for equal distribution)
    if len(counts) > 1:
        balance = min(counts) / max(counts)
        print(f"Balance ratio: {balance:.2f} (1.0 = perfect balance)")
    
    return interleaved


def preprocess_training_dataset(tokenizer, skill_data, pack=True, max_length: int = 768, drop_truncated: bool = True):
    """
    Preprocess training data, optionally filtering out truncated examples.
    
    Args:
        tokenizer: Tokenizer to use
        skill_data: List of text strings
        pack: Whether to use sequence packing
        max_length: Maximum sequence length
        drop_truncated: If True, drop examples that would be truncated (default: True)
    
    Returns:
        Dataset with preprocessed examples
    """
    # Filter out truncated examples if requested
    if drop_truncated:
        filtered_data = []
        dropped_count = 0
        for text in skill_data:
            token_length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            if token_length <= max_length:
                filtered_data.append(text)
            else:
                dropped_count += 1
        
        if dropped_count > 0:
            print(f"\n⚠️  Dropped {dropped_count}/{len(skill_data)} examples ({dropped_count/len(skill_data)*100:.1f}%) that exceeded max_length={max_length}")
            print(f"   Remaining examples: {len(filtered_data)}")
        
        skill_data = filtered_data
    
    if pack:
        input_ids, attn = pack_sequences(tokenizer, skill_data, max_length=max_length)
        dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attn, "labels": input_ids})
    else:
        tokenized_data = tokenizer(skill_data, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
        dataset = Dataset.from_dict({"input_ids": tokenized_data.input_ids, "attention_mask": tokenized_data.attention_mask, "labels": tokenized_data.input_ids})
    return dataset


def run_benchmark_suite(model, tokenizer, benchmark_names, max_samples=None, use_test_execution=True, log_prefix="", 
                       use_evalplus=True, evalplus_backend="hf", model_name=None):
    """
    Run multiple benchmarks on a model and return aggregated results.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        benchmark_names: List of benchmark names to run (e.g., ['humaneval', 'humanevalplus', 'mbpp'])
        max_samples: Maximum number of samples to evaluate per benchmark
        use_test_execution: If True, use actual test execution for Pass@1 (slower but accurate)
        log_prefix: Prefix for logging (e.g., "baseline", "final")
        use_evalplus: If True, use EvalPlus framework (default: True)
        evalplus_backend: Backend for EvalPlus ('hf', 'vllm', 'openai', etc.)
        model_name: Model name for EvalPlus (required when use_evalplus=True)
    
    Returns:
        dict: Contains results for each benchmark with metrics and samples
    """
    if use_evalplus:
        # Use EvalPlus framework for accurate evaluation
        if not model_name:
            print("⚠️  Warning: model_name required for EvalPlus. Falling back to legacy benchmarks.")
            use_evalplus = False
        else:
            print(f"\n{'='*80}")
            print(f"Using EvalPlus Framework (backend: {evalplus_backend})")
            print(f"Model: {model_name}")
            print(f"Benchmarks: {', '.join(benchmark_names)}")
            print(f"Max samples per benchmark: {max_samples or 'all'}")
            print(f"{'='*80}")
            
            all_results = {}
            
            # Map benchmark names to EvalPlus datasets
            dataset_map = {
                'humaneval': 'humaneval',
                'humanevalplus': 'humaneval',
                'mbpp': 'mbpp'
            }
            
            for benchmark_name in benchmark_names:
                if benchmark_name not in dataset_map:
                    print(f"⚠️  Warning: '{benchmark_name}' not supported by EvalPlus. Skipping.")
                    continue
                
                dataset = dataset_map[benchmark_name]
                print(f"\n--- Running {benchmark_name.upper()} with EvalPlus ---")
                
                # Initialize EvalPlus benchmark
                evalplus_bench = EvalPlusBenchmark(
                    tokenizer=tokenizer,
                    model_name=model_name,
                    dataset=dataset,
                    backend=evalplus_backend,
                    greedy=True
                )
                
                # Run benchmark
                result = evalplus_bench.run_benchmark(model=model, max_samples=max_samples)
                all_results[benchmark_name] = result
                
                # Print summary
                metrics = result.get('metrics', {})
                print(f"\n{benchmark_name.upper()} Results:")
                print(f"  Base Pass@1: {metrics.get('base_pass@1', 0.0):.2%}")
                print(f"  Plus Pass@1: {metrics.get('plus_pass@1', 0.0):.2%}")
            
            print(f"\n{'='*80}")
            print(f"EvalPlus Benchmark Suite Complete")
            print(f"{'='*80}\n")
            
            return all_results
    
    # Legacy benchmarks (fallback)
    # Initialize available benchmarks with fixed high token limits (no adaptive adjustment)
    available_benchmarks = {
        'humaneval': HumanEvalBenchmark(tokenizer, max_new_tokens=1024, timeout_seconds=10, use_test_execution=use_test_execution, use_adaptive_tokens=False),
        'humanevalplus': HumanEvalPlusBenchmark(tokenizer, max_new_tokens=1024, timeout_seconds=10, use_test_execution=use_test_execution, use_adaptive_tokens=False),
        'mbpp': MBPPBenchmark(tokenizer, max_new_tokens=1024, timeout_seconds=10, use_test_execution=use_test_execution, use_adaptive_tokens=False)
    }
    
    # Validate requested benchmarks
    for benchmark_name in benchmark_names:
        if benchmark_name not in available_benchmarks:
            print(f"⚠️  Warning: Unknown benchmark '{benchmark_name}'. Available: {list(available_benchmarks.keys())}")
            print(f"   Skipping '{benchmark_name}'")
    
    # Filter to only valid benchmarks
    valid_benchmarks = [b for b in benchmark_names if b in available_benchmarks]
    
    if not valid_benchmarks:
        print("❌ No valid benchmarks to run")
        return {}
    
    print(f"\n{'='*80}")
    print(f"Running Benchmark Suite: {', '.join(valid_benchmarks)}")
    if log_prefix:
        print(f"Stage: {log_prefix}")
    print(f"Max samples per benchmark: {max_samples or 'all'}")
    print(f"Test execution: {'enabled' if use_test_execution else 'disabled (heuristic only)'}")
    print(f"{'='*80}")
    
    all_results = {}
    
    for benchmark_name in valid_benchmarks:
        benchmark = available_benchmarks[benchmark_name]
        
        print(f"\n--- Running {benchmark_name.upper()} Benchmark ---")
        
        # Run benchmark
        result = benchmark.run_benchmark(
            model=model,
            max_samples=max_samples,
            log_to_wandb=False,  # We'll log manually to wandb in the calling function
            prefix=log_prefix
        )
        
        all_results[benchmark_name] = result
        
        # Print summary for this benchmark
        metrics = result['metrics']
        print(f"\n{benchmark_name.upper()} Results:")
        print(f"  Pass@1: {metrics.get('pass@1', 0.0):.2%}")
        print(f"  Syntax Score: {metrics.get('syntax_score', 0.0):.2%}")
        print(f"  Entry Point Score: {metrics.get('entry_point_score', 0.0):.2%}")
        if use_test_execution:
            print(f"  Tests Passed: {metrics.get('tests_passed', 0)}/{metrics.get('tests_run', 0)}")
        print(f"  Truncation Rate: {metrics.get('truncation_rate', 0.0):.2%}")
        print(f"  Avg Tokens Generated: {metrics.get('avg_tokens_generated', 0.0):.1f}")
        print(f"  Max Token Limits: min={metrics.get('min_token_limit', 0):.0f}, avg={metrics.get('avg_token_limit', 0.0):.0f}, max={metrics.get('max_token_limit', 0):.0f}")
    
    print(f"\n{'='*80}")
    print(f"Benchmark Suite Complete")
    print(f"{'='*80}\n")
    
    return all_results


def get_optimal_lr_config(args):
    """
    Determine optimal learning rate configuration based on training duration.
    """
    num_epochs = args.num_epochs
    
    # Auto strategy selection
    if args.lr_strategy == "auto":
        if num_epochs == 1:
            strategy = "constant"
            warmup_ratio = 0.05  # Very short warmup for 1 epoch
            learning_rate = 3e-5  # Slightly lower LR for stability
        elif num_epochs <= 3:
            strategy = "linear"
            warmup_ratio = 0.1 if not args.scale_lr_for_short_training else 0.05
            learning_rate = 4e-5 if args.scale_lr_for_short_training else 5e-5
        else:
            strategy = "cosine_with_restarts" if args.cosine_restarts else "cosine"
            warmup_ratio = 0.1
            learning_rate = 5e-5
    else:
        # Use user-specified strategy
        strategy = args.lr_strategy
        if args.scale_lr_for_short_training and num_epochs <= 3:
            warmup_ratio = 0.05
            learning_rate = 3e-5 if num_epochs == 1 else 4e-5
        else:
            warmup_ratio = 0.1
            learning_rate = 5e-5
    
    # Handle cosine_restarts override
    if args.cosine_restarts and strategy in ["cosine", "auto"]:
        strategy = "cosine_with_restarts"
    
    # Configure scheduler kwargs
    scheduler_kwargs = {}
    if strategy == "cosine_with_restarts":
        # Adjust cycles based on training length
        if num_epochs == 1:
            scheduler_kwargs = {"num_cycles": 0.5}  # Half cycle for 1 epoch
        elif num_epochs <= 3:
            scheduler_kwargs = {"num_cycles": 1}    # One cycle for short training
        else:
            scheduler_kwargs = {"num_cycles": 2}    # Default 2 cycles
    
    print(f"📊 Learning Rate Configuration:")
    print(f"   Strategy: {strategy}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Warmup Ratio: {warmup_ratio}")
    if scheduler_kwargs:
        print(f"   Scheduler Args: {scheduler_kwargs}")
    print()
    
    return strategy, learning_rate, warmup_ratio, scheduler_kwargs


def main(args):
    # 0. Check environment variables first
    hf_token, wandb_token = check_environment_variables()
    
    # 1. Initialize services (HuggingFace Hub and Weights & Biases)
    initialize_services(hf_token, wandb_token)
    
    # 2. Initialize wandb project
    wandb.init(project="dylo-moe-full-training")

    # 3. Download checkpoint from W&B if resuming
    checkpoint_path = None
    if args.wandb_checkpoint_artifact:
        print(f"\n--- Downloading checkpoint from W&B artifact: {args.wandb_checkpoint_artifact} ---")
        try:
            artifact = wandb.use_artifact(args.wandb_checkpoint_artifact, type='model')
            artifact_dir = artifact.download()
            checkpoint_path = artifact_dir
            print(f"✓ Checkpoint downloaded to: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Failed to download W&B artifact: {e}")
            print("Proceeding without checkpoint...")
            checkpoint_path = None

    # 4. Instantiate the model (using hf_token from environment check)
    model = DyLoRA_MoE(
        args.model_name,
        num_experts=args.num_experts,
        token=hf_token,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        balance_coefficient=args.balance_coefficient  # Load balancing auxiliary loss coefficient
    )
    
    print(f"Initialized {model.expert_manager.num_experts} experts")
    print(f"⚠️  Router will use DENSE (softmax) routing during training to prevent expert collapse")
    print(f"   All experts will receive gradients. Sparse (top-k) routing only used at inference.")
    
    # Ensure all LoRA parameters are trainable
    print("\n--- Verifying trainable parameters ---")
    lora_params = 0
    router_params = 0
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            if "lora" in name.lower():
                lora_params += param.numel()
            elif "router" in name.lower() or "gate" in name.lower():
                router_params += param.numel()
        else:
            frozen_params += param.numel()
    
    print(f"LoRA parameters: {lora_params:,} (trainable)")
    print(f"Router parameters: {router_params:,} (trainable)")
    print(f"Frozen parameters: {frozen_params:,} (includes base model + lm_head)")
    print(f"Total parameters: {total_params:,}")
    print(f"Total trainable: {lora_params + router_params:,}")
    print(f"Trainable %: {(lora_params + router_params) / total_params * 100:.2f}%")
    
    # Verify weight sharing: frozen params should be counted once regardless of num_experts
    print(f"\n--- Memory Efficiency Verification ---")
    print(f"Number of experts: {model.expert_manager.num_experts}")
    print(f"LoRA params per expert (approx): {lora_params // model.expert_manager.num_experts:,}")
    print(f"Base model params (shared): {frozen_params:,}")
    print(f"✓ All experts share the same {frozen_params:,} frozen base weights")
    print(f"✓ Only {lora_params:,} adapter params differ between experts")
    print(f"✓ LM head is frozen (standard LoRA practice)")
    
    if lora_params == 0:
        raise ValueError("No LoRA parameters are trainable! Check model initialization.")

    # 3. Create tokenizer and data stream
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Load datasets based on --datasets argument
    # Default: code_alpaca,mbpp
    # Available: code_alpaca, mbpp, evol_instruct, code_feedback, python_codes_25k, 
    #            python_code_instructions_18k, python_code_23k_sharegpt
    
    dataset_names = args.datasets.split(',')
    print(f"\n--- Loading Datasets: {', '.join(dataset_names)} ---")
    print(f"Available datasets: {', '.join(AVAILABLE_DATASETS.keys())}")
    
    # Load each dataset
    loaded_datasets = []
    for dataset_name in dataset_names:
        dataset_name = dataset_name.strip()
        if dataset_name == 'humaneval':
            print(f"⚠️  Skipping {dataset_name} (evaluation only, not for training)")
            continue
        if dataset_name not in AVAILABLE_DATASETS:
            print(f"⚠️  Unknown dataset: {dataset_name}. Skipping.")
            continue
        try:
            dataset = get_dataset(dataset_name, with_validation=True)
            loaded_datasets.append((dataset_name, dataset))
            print(f"✓ Loaded {dataset_name}: {len(dataset['train'])} train, {len(dataset['validation'])} val")
        except Exception as e:
            print(f"⚠️  Error loading {dataset_name}: {e}")
    
    if not loaded_datasets:
        raise ValueError("No datasets were successfully loaded. Please check --datasets argument.")
    
    if args.benchmark_strategy != 'none':
    # Download HumanEval for evaluation only (not used in training)
        humaneval_dataset = download_humaneval()
    
    # For backwards compatibility, assign first two datasets as primary datasets
    # (used in interleaved sampling if enabled)
    if len(loaded_datasets) >= 1:
        primary_dataset_name, primary_dataset = loaded_datasets[0]
        code_alpaca_dataset = primary_dataset  # For backwards compatibility
    if len(loaded_datasets) >= 2:
        secondary_dataset_name, secondary_dataset = loaded_datasets[1]
        mbpp_dataset = secondary_dataset  # For backwards compatibility
    
    # Display device information before training
    print_device_info(force_device=args.device)
    training_device = get_device(args.device)
    if args.device:
        print(f"✓ Using user-specified device: {training_device.upper()}\n")
    else:
        print(f"✓ Auto-detected optimal device: {training_device.upper()}\n")
    
    # Move model to GPU if available (needed for baseline evaluation)
    model = move_model_to_device(model, verbose=True, force_device=args.device)
    
    # Baseline: Evaluate base model (before LoRA training) on benchmarks
    # Only run if benchmark_strategy is not 'none'
    if args.benchmark_strategy != 'none':
        print("\n" + "="*80)
        print("BASELINE EVALUATION: Base Model (Before LoRA Training)")
        print("="*80)
        
        # Run benchmark suite on base model
        baseline_results = run_benchmark_suite(
            model=model,
            tokenizer=tokenizer,
            benchmark_names=args.benchmarks,
            max_samples=None,  # Run on full benchmark datasets
            use_test_execution=True,
            log_prefix="baseline",
            use_evalplus=args.use_evalplus,
            evalplus_backend=args.evalplus_backend,
            model_name=args.model_name
        )
        
        # Log results to wandb for each benchmark
        for benchmark_name, result in baseline_results.items():
            metrics = result['metrics']
            wandb.log({
                f"baseline/{benchmark_name}_pass@1": metrics.get('pass@1', 0.0),
                f"baseline/{benchmark_name}_syntax_score": metrics.get('syntax_score', 0.0),
                f"baseline/{benchmark_name}_entry_point_score": metrics.get('entry_point_score', 0.0),
                f"baseline/{benchmark_name}_truncation_rate": metrics.get('truncation_rate', 0.0),
                f"baseline/{benchmark_name}_avg_tokens_generated": metrics.get('avg_tokens_generated', 0.0),
                f"baseline/{benchmark_name}_tests_passed": metrics.get('tests_passed', 0),
                f"baseline/{benchmark_name}_tests_run": metrics.get('tests_run', 0),
            }, commit=False)
        
        print("\n✓ Baseline evaluation complete")
        print("="*80 + "\n")
    else:
        print("\nℹ Skipping baseline evaluation (benchmark_strategy='none')\n")
    
    # Prepare training data from loaded datasets
    print("\n--- Preparing Training Data from Loaded Datasets ---")
    
    # Extract text from each dataset based on their schema
    def extract_text_from_dataset(dataset_name: str, dataset_dict):
        """Extract training text from dataset based on its schema."""
        train_data = []
        dataset = dataset_dict['train']
        
        # Get first example to inspect schema
        first_example = dataset[0]
        
        # Handle different dataset formats
        if 'instruction' in first_example and 'output' in first_example:
            # Alpaca-style format (code_alpaca, evol_instruct, python_code_instructions_18k, python_codes_25k)
            for ex in dataset:
                text = ex['instruction']
                if ex.get('input'):  # Some datasets have input field
                    text += "\n" + ex['input']
                text += "\n" + ex['output']
                train_data.append(text)
        elif 'query' in first_example and 'answer' in first_example:
            # CodeFeedback format
            for ex in dataset:
                train_data.append(ex['query'] + "\n" + ex['answer'])
        elif 'conversations' in first_example:
            # ShareGPT format (python_code_23k_sharegpt)
            for ex in dataset:
                # Concatenate conversation turns
                text = "\n".join([turn.get('value', '') for turn in ex['conversations']])
                train_data.append(text)
        elif 'text' in first_example:
            # MBPP format or simple text format
            train_data = [ex['text'] for ex in dataset]
        else:
            raise ValueError(f"Unknown dataset format for {dataset_name}. Fields: {list(first_example.keys())}")
        
        return train_data
    
    # Process all loaded datasets
    all_dataset_data = []
    for dataset_name, dataset_dict in loaded_datasets:
        data = extract_text_from_dataset(dataset_name, dataset_dict)
        all_dataset_data.append((dataset_name, data))
        print(f"✓ {dataset_name}: {len(data)} examples")
    
    # Choose sampling strategy based on CLI flag
    if args.interleaved_sampling and len(all_dataset_data) == 2:
        print("\n--- Using Interleaved Sampling (Equal 50/50 representation) ---")
        # Use interleaved sampling for equal representation (works with 2 datasets)
        skill1_name, skill1_data = all_dataset_data[0]
        skill2_name, skill2_data = all_dataset_data[1]
        combined_data = create_interleaved_dataset(skill1_data, skill2_data)
        print(f"Domain representation: ~50% {skill1_name}, ~50% {skill2_name} (balanced)")
    else:
        print("\n--- Using Concatenation (dataset size based) ---")
        # Concatenate all datasets
        combined_data = []
        for dataset_name, data in all_dataset_data:
            combined_data.extend(data)
        
        # Print distribution
        total = len(combined_data)
        print("Dataset distribution:")
        for dataset_name, data in all_dataset_data:
            print(f"  {dataset_name}: {len(data)} ({len(data)/total*100:.1f}%)")
    
    print(f"\nTotal training examples: {len(combined_data)}")
    
    if args.training_subset:
        subset_size = int(len(combined_data) * (args.training_subset / 100))
        combined_data = combined_data[:subset_size]
    
    # Create train/validation split (90/10) from combined data for proper evaluation
    # HumanEval will be used for benchmarking only, not for per-epoch evaluation
    train_size = int(len(combined_data) * 0.9)
    train_data = combined_data[:train_size]
    val_data = combined_data[train_size:]
    
    print(f"\nCombined dataset split:")
    print(f"  Training: {len(train_data)} examples ({len(train_data)/len(combined_data)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} examples ({len(val_data)/len(combined_data)*100:.1f}%)")
    if args.benchmark_strategy != 'none':
        print(f"  HumanEval (benchmark only): {len(humaneval_dataset)} examples")




    # 5. Configure training arguments with optimal learning rate strategy
    lr_strategy, learning_rate, warmup_ratio, scheduler_kwargs = get_optimal_lr_config(args)
    
    # Updated training schedule for Phase 1 improvements:
    # - Adaptive learning rate strategy based on training duration
    # - Using cosine_with_restarts scheduler for better exploration
    # - Reduced early stopping patience to 2 (from 3) to stop at optimal point
    # - Added early stopping threshold of 0.005 for plateau detection
    training_args = TrainingArguments(
        output_dir="./results_full",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Configurable: effective batch size = train_batch_size * gradient_accumulation_steps
        gradient_checkpointing=False,  # DISABLED: Breaks gradient flow with multi-expert routing
        learning_rate=learning_rate,  # Adaptive based on training duration
        weight_decay=args.weight_decay,  # Configurable L2 regularization
        warmup_ratio=warmup_ratio,  # Adaptive based on training duration
        lr_scheduler_type=lr_strategy,
        lr_scheduler_kwargs=scheduler_kwargs,
        fp16=args.fp16,
        bf16=args.bf16,
        label_smoothing_factor=args.label_smoothing,  # Prevent overfitting by smoothing targets
        logging_dir='./logs_full',
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        save_safetensors=False,  # CRITICAL: Prevent Trainer from saving model.safetensors (11GB merged model)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Track validation loss to prevent overfitting
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=True,
    )

    # 6. Instantiate the trainer
    
    # Create training and validation datasets from the split
    print(f"\n--- Preparing Training and Validation Datasets ---")
    
    # Analyze truncation statistics before tokenization
    print(f"\n{'='*80}")
    print("TRUNCATION ANALYSIS")
    print(f"{'='*80}")
    train_stats = analyze_truncation_stats(train_data, tokenizer, max_length=args.max_seq_length, dataset_name="Training Set")
    val_stats = analyze_truncation_stats(val_data, tokenizer, max_length=args.max_seq_length, dataset_name="Validation Set")
    
    # Log truncation stats to wandb
    wandb.log({
        "data_prep/train_truncation_rate": train_stats['truncation_rate'],
        "data_prep/train_avg_length": train_stats['avg_length'],
        "data_prep/train_p95_length": train_stats['p95'],
        "data_prep/val_truncation_rate": val_stats['truncation_rate'],
        "data_prep/val_avg_length": val_stats['avg_length'],
        "data_prep/val_p95_length": val_stats['p95'],
    }, commit=False)
    
    print(f"{'='*80}\n")
    
    # Preprocess datasets - by default, drop truncated examples unless --keep_truncated is set
    drop_truncated = not args.keep_truncated
    if drop_truncated:
        print("✓ Truncated examples will be DROPPED (use --keep_truncated to keep them)")
    else:
        print("⚠️  Truncated examples will be KEPT and truncated (may result in incomplete training data)")
    
    train_dataset = preprocess_training_dataset(tokenizer, train_data, pack=True, max_length=args.max_seq_length, drop_truncated=drop_truncated)
    val_dataset = preprocess_evaluation_dataset(tokenizer, Dataset.from_dict({"text": val_data}), max_length=args.max_seq_length)
    
    print(f"Training dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    
    # Exit early if --data_prep_only flag is set
    if args.data_prep_only:
        print(f"\n{'='*80}")
        print("DATA PREPARATION COMPLETE")
        print(f"{'='*80}")
        print("\n✅ Data preparation completed successfully!")
        print(f"   Training examples: {len(train_dataset)}")
        print(f"   Validation examples: {len(val_dataset)}")
        print(f"\n📊 Truncation Statistics Summary:")
        print(f"   Training Set:")
        print(f"      - Truncation Rate: {train_stats['truncation_rate']:.2%}")
        print(f"      - Avg Length: {train_stats['avg_length']:.1f} tokens")
        print(f"      - P95 Length: {train_stats['p95']:.0f} tokens")
        print(f"   Validation Set:")
        print(f"      - Truncation Rate: {val_stats['truncation_rate']:.2%}")
        print(f"      - Avg Length: {val_stats['avg_length']:.1f} tokens")
        print(f"      - P95 Length: {val_stats['p95']:.0f} tokens")
        print(f"\n💡 To proceed with training, run without --data_prep_only flag")
        print(f"{'='*80}\n")
        
        # Finish wandb run
        wandb.finish()
        return  # Exit main() function
    
    # Prepare HumanEval for benchmarking only (not for per-epoch evaluation)
    if args.benchmark_strategy != 'none':
        print(f"\n--- Preparing HumanEval Benchmark Dataset ---")
        humaneval_eval_data = []
        for ex in humaneval_dataset:
            prompt = ex.get('prompt', '')
            if prompt:
                humaneval_eval_data.append(prompt)
        
        humaneval_eval_dataset = Dataset.from_dict({"text": humaneval_eval_data})
        humaneval_benchmark = preprocess_evaluation_dataset(tokenizer, humaneval_eval_dataset)
        
        print(f"HumanEval benchmark: {len(humaneval_benchmark)} examples")

    
    # Create callbacks with updated early stopping parameters
    # Early stopping can be disabled via --disable_early_stopping flag
    callbacks = [
        GradientMonitoringCallback(model, num_experts=model.expert_manager.num_experts),
        DyLoRAMonitoringCallback(model, num_experts=model.expert_manager.num_experts),
        PeftCheckpointCallback(num_experts=model.expert_manager.num_experts),  # Save PEFT adapters only (prevents full model saving)
    ]
    
    print(f"✓ PEFT-only checkpoint saving enabled")
    print(f"  • Checkpoints will contain PEFT adapters + router state ONLY")
    print(f"  • Full merged model weights will NOT be saved (saves disk space)")
    print(f"  • Use benchmark.py to load PEFT checkpoints for evaluation")
    
    # Add benchmark callback if strategy is 'epoch'
    if args.benchmark_strategy == 'epoch':
        callbacks.append(BenchmarkCallback(
            model=model,
            tokenizer=tokenizer,
            benchmark_names=args.benchmarks,
            use_test_execution=True,
            use_evalplus=args.use_evalplus,
            evalplus_backend=args.evalplus_backend,
            model_name=args.model_name
        ))
        print(f"✓ Epoch benchmarking enabled - will run {args.benchmarks} after each epoch")
    
    # Conditionally add early stopping callback
    if not args.disable_early_stopping:
        callbacks.insert(0, EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.005  # Minimum improvement threshold for plateau detection
        ))
        print(f"✓ Early stopping enabled (patience={args.early_stopping_patience}, threshold=0.005)")
    else:
        print("ℹ Early stopping disabled - will train for all epochs")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Use validation split for per-epoch evaluation and early stopping
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=callbacks,
    )

    # 7. Train the model on the combined dataset
    print("\n--- Training on Combined Dataset ---")
    print(f"Training with {model.expert_manager.num_experts} experts")
    
    # Use W&B checkpoint if downloaded, otherwise use legacy resume flag
    resume_checkpoint = checkpoint_path if checkpoint_path else (True if args.resume_from_checkpoint else None)
    if checkpoint_path:
        print(f"Resuming from W&B checkpoint: {checkpoint_path}")
    elif args.resume_from_checkpoint:
        print("Resuming from local checkpoint (legacy mode)")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 8. Final evaluation on Benchmark Suite (after training)
    # Only run if benchmark_strategy is 'final' or 'epoch' 
    if args.benchmark_strategy in ['final', 'epoch']:
        print("\n" + "="*80)
        print("FINAL EVALUATION: Trained Model (After LoRA Training)")
        print("="*80)
        
        # Run benchmark suite on trained model
        final_results = run_benchmark_suite(
            model=model,
            tokenizer=tokenizer,
            benchmark_names=args.benchmarks,
            max_samples=None,  # Run on full benchmark datasets
            use_test_execution=True,
            log_prefix="final",
            use_evalplus=args.use_evalplus,
            evalplus_backend=args.evalplus_backend,
            model_name=args.model_name
        )
        
        # Log results to wandb for each benchmark
        for benchmark_name, result in final_results.items():
            metrics = result['metrics']
            wandb.log({
                f"final/{benchmark_name}_pass@1": metrics.get('pass@1', 0.0),
                f"final/{benchmark_name}_syntax_score": metrics.get('syntax_score', 0.0),
                f"final/{benchmark_name}_entry_point_score": metrics.get('entry_point_score', 0.0),
                f"final/{benchmark_name}_truncation_rate": metrics.get('truncation_rate', 0.0),
                f"final/{benchmark_name}_avg_tokens_generated": metrics.get('avg_tokens_generated', 0.0),
                f"final/{benchmark_name}_tests_passed": metrics.get('tests_passed', 0),
                f"final/{benchmark_name}_tests_run": metrics.get('tests_run', 0),
            }, commit=False)
        
        print("\n✓ Final evaluation complete")
    else:
        print("\nℹ Skipping final evaluation (benchmark_strategy='none')\n")
    
    # Also evaluate loss on the first benchmark dataset for compatibility
    if args.benchmark_strategy != 'none' and args.benchmarks and len(args.benchmarks) > 0:
        first_benchmark = args.benchmarks[0]
        # Map benchmark names to dataset variables
        benchmark_dataset_map = {
            'humaneval': humaneval_benchmark,
            'humanevalplus': humaneval_benchmark,  # HumanEval+ uses same underlying data
            'mbpp': get_dataset('mbpp', with_validation=True)['validation']  # Use MBPP validation
        }
        
        if first_benchmark in benchmark_dataset_map:
            trainer.eval_dataset = benchmark_dataset_map[first_benchmark]
            final_eval = trainer.evaluate(metric_key_prefix="eval")
            print(f"\nFinal {first_benchmark.upper()} Loss: {final_eval['eval_loss']:.4f}")
            wandb.log({
                f"final/{first_benchmark}_loss": final_eval['eval_loss'],
            }, commit=False)
    
        print("\n✓ Final evaluation complete")
        print("="*80 + "\n")

    # 9. Save the best model, trainer state, and DyLoRA-MoE state
    print("\n--- Saving Best Model and Full State ---")
    best_model_dir = "./results_full/best_model"
    
    # CRITICAL: Save PEFT adapters separately (NOT merged) to preserve MoE capability
    # This saves each expert's LoRA adapters as separate files (expert_0.pt, expert_1.pt, etc.)
    # Without this, the model collapses to a single merged model with no MoE routing
    print("\n1. Saving PEFT expert adapters (separate files for each expert)...")
    peft_adapters_dir = os.path.join(best_model_dir, "peft_adapters")
    save_lora_experts(model, peft_adapters_dir)
    print(f"✓ PEFT adapters saved to {peft_adapters_dir}")
    
    # Save DyLoRA-MoE specific state (router and skill library)
    print("\n2. Saving DyLoRA-MoE state (router + skill library)...")
    dylo_moe_state_dir = os.path.join(best_model_dir, "dylo_moe_state")
    save_dylo_moe_state(model, dylo_moe_state_dir)
    print(f"✓ DyLoRA-MoE state saved to {dylo_moe_state_dir}")
    
    # Save tokenizer and training arguments
    print("\n3. Saving tokenizer and training config...")
    tokenizer.save_pretrained(best_model_dir)
    trainer.save_state()  # Saves optimizer, scheduler, etc. to output_dir
    torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
    print(f"✓ Tokenizer and training args saved")
    
    # Save config.json with DyLoRA-MoE metadata for model loading
    print("\n4. Creating config.json with DyLoRA-MoE metadata...")
    base_config = AutoConfig.from_pretrained(args.model_name)
    base_config_dict = base_config.to_dict()
    
    # Add DyLoRA-specific metadata to enable proper loading
    base_config_dict["model_type"] = "dylora-moe"  # Mark as DyLoRA-MoE format
    base_config_dict["base_model_name_or_path"] = args.model_name
    base_config_dict["_dylora_num_experts"] = args.num_experts
    base_config_dict["_dylora_lora_r"] = args.lora_r
    base_config_dict["_dylora_lora_alpha"] = args.lora_alpha
    base_config_dict["_dylora_lora_dropout"] = args.lora_dropout
    base_config_dict["_dylora_trained"] = True
    base_config_dict["_dylora_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    base_config_dict["_dylora_format"] = "peft_separated"  # Indicates separate PEFT files
    
    config_path = os.path.join(best_model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(base_config_dict, f, indent=2)
    print(f"✓ Configuration saved to {config_path} (format: dylora-moe with separated PEFT adapters)")
    
    print(f"\n{'='*80}")
    print("CHECKPOINT STRUCTURE:")
    print(f"{'='*80}")
    print(f"📁 {best_model_dir}/")
    print(f"   ├── config.json                    (DyLoRA-MoE metadata)")
    print(f"   ├── tokenizer files                (tokenizer.json, etc.)")
    print(f"   ├── 📁 peft_adapters/")
    print(f"   │   ├── expert_0.pt                (Expert 0 LoRA weights)")
    print(f"   │   ├── expert_1.pt                (Expert 1 LoRA weights)")
    print(f"   │   ├── expert_2.pt                (Expert 2 LoRA weights)")
    print(f"   │   └── expert_3.pt                (Expert 3 LoRA weights)")
    print(f"   └── 📁 dylo_moe_state/")
    print(f"       └── router.pt                  (Router state)")
    print(f"{'='*80}")
    print(f"\n✓ Model saved in PEFT-separated format (MoE routing preserved)")
    print(f"✓ Base model ({args.model_name}) will be loaded separately during inference")
    print(f"{'='*80}\n")

    # 10. Upload the entire output directory as a wandb artifact
    print("--- Uploading Artifacts to W&B ---")
    artifact = wandb.Artifact('best-dylora-model-full', type='model')
    artifact.add_dir(training_args.output_dir)
    wandb.log_artifact(artifact)
    print("Best model, trainer state, and DyLoRA-MoE state saved and uploaded to wandb.")

    # 12. Print the final model architecture and trainable parameters
    print("--- Final Model Architecture ---")
    print(model)
    print("--- Trainable Parameters ---")
    print_trainable_parameters(model)

    wandb.finish()

def parse_args(argv=None):
    """Parse command line arguments. Can be called with custom argv for programmatic use."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint (legacy - local files).")
    parser.add_argument("--wandb_checkpoint_artifact", type=str, default=None, 
                        help="W&B artifact path to resume from (e.g., 'johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0'). "
                             "Downloads the checkpoint from W&B and resumes training. Overrides --resume_from_checkpoint.")
    parser.add_argument("--training_subset", type=int, default=None, help="Percentage of training data to use.")
    parser.add_argument("--eval_subset", type=int, default=None, help="Percentage of evaluation data to use.")
    parser.add_argument("--model_name", type=str, default="google/codegemma-2b", help="The base model to use.")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of LoRA experts to create (fixed at initialization).")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r value.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout value (increased to 0.1 to prevent overfitting).")
    parser.add_argument("--balance_coefficient", type=float, default=0.15, help="Coefficient for load balancing auxiliary loss (0.1-0.5 recommended to prevent expert collapse; 0 to disable).")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay (L2 regularization) coefficient to prevent overfitting (increased to 0.05 for better generalization).")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor to prevent overfitting (default: 0.1, range: 0.0-0.3 for stronger regularization).")
    parser.add_argument("--interleaved_sampling", action="store_true", help="Use interleaved sampling for balanced dataset representation (50/50 Code Alpaca and MBPP).")
    parser.add_argument("--cosine_restarts", action="store_true", help="Use cosine_with_restarts learning rate scheduler with 2 cycles for better exploration.")
    parser.add_argument("--lr_strategy", type=str, default="auto", choices=["auto", "constant", "linear", "cosine", "cosine_restarts"], 
                        help="Learning rate strategy. 'auto' adapts based on num_epochs: constant for 1 epoch, linear for 2-3, cosine for 4+.")
    parser.add_argument("--scale_lr_for_short_training", action="store_true", 
                        help="Scale learning rate and warmup for short training runs (< 4 epochs).")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Per-device training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Per-device evaluation batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps (effective batch size = train_batch_size * gradient_accumulation_steps).")
    parser.add_argument("--logging_steps", type=int, default=50, help="Number of steps between logging metrics to wandb and console.")
    parser.add_argument("--max_seq_length", type=int, default=2048, 
                        help="Maximum sequence length for tokenization (default: 2048, CodeGemma supports up to 8192)")
    parser.add_argument("--keep_truncated", action="store_true", 
                        help="Keep examples that exceed max_seq_length (they will be truncated). By default, truncated examples are dropped to avoid training on incomplete data.")
    parser.add_argument("--disable_early_stopping", action="store_true", help="Disable early stopping and train for all epochs.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs with no improvement before stopping (only applies if early stopping is enabled).")
    parser.add_argument(
        "--datasets", 
        type=str, 
        default="code_alpaca,mbpp", 
        help="Comma-separated list of datasets to use for training. "
             "Available: code_alpaca, mbpp, evol_instruct, code_feedback, python_codes_25k, "
             "python_code_instructions_18k, python_code_23k_sharegpt. "
             "Example: --datasets 'code_alpaca,mbpp,evol_instruct'"
    )
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["humanevalplus"],
                       help="Benchmarks to run for baseline and final evaluation: humaneval, humanevalplus, mbpp (default: humanevalplus)")
    parser.add_argument("--benchmark-strategy", type=str, default="final", choices=["none", "epoch", "final"],
                       help="Benchmarking strategy: 'none' (no benchmarks), 'epoch' (run after each epoch), 'final' (only baseline and final, default)")
    parser.add_argument("--use_evalplus", action="store_true", default=True,
                       help="Use EvalPlus framework for benchmarking (default: True, more accurate)")
    parser.add_argument("--no_evalplus", action="store_false", dest="use_evalplus",
                       help="Disable EvalPlus and use legacy benchmarks instead")
    parser.add_argument("--evalplus_backend", type=str, default="hf", choices=["hf", "vllm", "openai", "anthropic", "google"],
                       help="Backend for EvalPlus evaluation (default: hf)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for training and inference: 'cpu', 'cuda', 'cuda:0', 'mps', etc. (default: auto-detect)")
    parser.add_argument("--data_prep_only", action="store_true",
                       help="Only perform data preparation and report statistics (including truncation analysis), then exit without training.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args)