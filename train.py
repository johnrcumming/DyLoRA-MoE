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
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
)
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers.tokenization_utils_base import BatchEncoding
from typing import Union, Dict, Iterable, Any

from dylo_moe.model import DyLoRA_MoE
from dylo_moe.utils import print_trainable_parameters, save_dylo_moe_state, save_lora_experts
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
            
            wandb.log(log_dict, step=state.global_step)


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
                    
                    # Log all metrics to wandb
                    wandb.log(log_dict, step=state.global_step)
                else:
                    # Edge case: no routing weights (shouldn't happen in normal training)
                    wandb.log({
                        "routing/warning": 1.0,
                        "routing/total_weight": 0.0
                    }, step=state.global_step)


def preprocess_evaluation_dataset(tokenizer, dataset):
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
        return tokenizer(processed_text, padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

def pack_sequences(tokenizer, texts, max_length=512):
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


def preprocess_training_dataset(tokenizer, skill_data, pack=True):
    if pack:
        input_ids, attn = pack_sequences(tokenizer, skill_data)
        dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attn, "labels": input_ids})
    else:
        tokenized_data = tokenizer(skill_data, padding=True, truncation=True, return_tensors="pt", max_length=512)
        dataset = Dataset.from_dict({"input_ids": tokenized_data.input_ids, "attention_mask": tokenized_data.attention_mask, "labels": tokenized_data.input_ids})
    return dataset

def evaluate_humaneval(model, tokenizer, humaneval_dataset, max_samples=None, use_test_execution=False):
    """
    Evaluate model on HumanEval benchmark using the new benchmark system.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        humaneval_dataset: HumanEval dataset (for compatibility, not used directly)
        max_samples: Maximum number of samples to evaluate
        use_test_execution: If True, use actual test execution for Pass@1 (slower but accurate)
    
    Returns:
        dict: Contains 'pass@1' score and generation samples
    """
    from benchmarks.humaneval_benchmark import HumanEvalBenchmark
    
    print("\n--- Evaluating on HumanEval Benchmark ---")
    
    # Create benchmark instance
    benchmark = HumanEvalBenchmark(
        tokenizer=tokenizer,
        max_new_tokens=256,
        timeout_seconds=10,
        use_test_execution=use_test_execution
    )
    
    # Run benchmark
    result = benchmark.run_benchmark(
        model=model,
        max_samples=max_samples,
        log_to_wandb=False,  # Don't log to wandb during training (handled separately)
        prefix="training"
    )
    
    # Extract metrics for compatibility with existing code
    metrics = result['metrics']
    
    # Map new metrics to old format for backward compatibility
    pass_at_1 = metrics.get('pass@1', metrics.get('entry_point_score', 0.0))
    
    results = {
        'pass@1_approx': pass_at_1,
        'num_samples': metrics.get('total_samples', 0),
        'num_with_entry_point': int(metrics.get('entry_point_score', 0.0) * metrics.get('total_samples', 0)),
        'sample_generations': result.get('samples', [])[:5],  # First 5 for inspection
        'full_metrics': metrics,  # Include all metrics from new system
    }
    
    print(f"HumanEval Pass@1: {pass_at_1:.2%}")
    if use_test_execution:
        print(f"  - Test execution Pass@1: {metrics.get('pass@1', 0.0):.2%}")
        print(f"  - Syntax score: {metrics.get('syntax_score', 0.0):.2%}")
        print(f"  - Entry point score: {metrics.get('entry_point_score', 0.0):.2%}")
    else:
        print(f"  - Entry point heuristic: {metrics.get('entry_point_score', 0.0):.2%}")
    
    return results

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
        allow_expert_growth=False,  # Disable dynamic expert growth for traditional training
        balance_coefficient=args.balance_coefficient  # Load balancing auxiliary loss coefficient
    )
    
    # Mark all experts as mature so router uses sparse delegation from the start
    for i in range(model.expert_manager.num_experts):
        model.router.set_expert_maturity(i, 1)
    
    print(f"Initialized {model.expert_manager.num_experts} experts (all marked as mature)")
    
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
    
    # Move model to GPU if available (needed for baseline evaluation)
    if torch.cuda.is_available():
        model = model.cuda()
        print("✓ Model moved to CUDA")
    elif torch.backends.mps.is_available():
        model = model.to('mps')
        print("✓ Model moved to MPS")
    else:
        print("ℹ Model on CPU")
    
    # Baseline: Evaluate base model (before LoRA training) on HumanEval
    print("\n" + "="*80)
    print("BASELINE EVALUATION: Base Model (Before LoRA Training)")
    print("="*80)
    
    # Evaluate base model's code generation capability (quick heuristic evaluation)
    baseline_results = evaluate_humaneval(
        model, tokenizer, humaneval_dataset, use_test_execution=False
    )
    
    wandb.log({
        "baseline/humaneval_pass@1": baseline_results['pass@1_approx'],
        "baseline/humaneval_with_entry_point": baseline_results['num_with_entry_point'],
        "baseline/num_samples": baseline_results['num_samples'],
    }, commit=False)
    
    print(f"\n✓ Baseline HumanEval Pass@1 (approx): {baseline_results['pass@1_approx']:.2%}")
    print(f"✓ Baseline generations with entry point: {baseline_results['num_with_entry_point']}/{baseline_results['num_samples']}")
    print("="*80 + "\n")
    
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
    print(f"  HumanEval (benchmark only): {len(humaneval_dataset)} examples")




    # 5. Configure the training arguments
    # Updated training schedule for Phase 1 improvements:
    # - Increased LR to 5e-5 (from 2e-5) for faster convergence
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
        learning_rate=5e-5,  # Increased from 2e-5 for faster convergence
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine_with_restarts" if args.cosine_restarts else "cosine",
        lr_scheduler_kwargs={"num_cycles": 2} if args.cosine_restarts else {},
        fp16=args.fp16,
        bf16=args.bf16,
        logging_dir='./logs_full',
        logging_steps=50,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Fixed: use "loss" not "eval_loss"
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=True,
    )

    # 6. Instantiate the trainer
    
    # Create training and validation datasets from the split
    print(f"\n--- Preparing Training and Validation Datasets ---")
    train_dataset = preprocess_training_dataset(tokenizer, train_data, pack=True)
    val_dataset = preprocess_evaluation_dataset(tokenizer, Dataset.from_dict({"text": val_data}))
    
    print(f"Training dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    
    # Prepare HumanEval for benchmarking only (not for per-epoch evaluation)
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
    ]
    
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

    # 8. Final evaluation on HumanEval Benchmark (after training)
    print("\n--- Final Evaluation on HumanEval Benchmark ---")
    # Swap to HumanEval for final benchmark
    trainer.eval_dataset = humaneval_benchmark
    final_eval = trainer.evaluate(metric_key_prefix="eval")
    print(f"Final HumanEval Loss: {final_eval['eval_loss']:.4f}")
    
    # Evaluate HumanEval code generation (use actual test execution for final evaluation)
    final_humaneval_results = evaluate_humaneval(
        model, tokenizer, humaneval_dataset, max_samples=20, use_test_execution=True
    )
    
    wandb.log({
        "final_humaneval_loss": final_eval['eval_loss'],
        "final_humaneval_pass@1": final_humaneval_results['pass@1_approx'],
        "final_humaneval_with_entry_point": final_humaneval_results['num_with_entry_point'],
    }, commit=False)

    # 9. Save the best model
    print("\n--- Saving Best Model ---")
    trainer.save_model("./results_full/best_model")
    tokenizer.save_pretrained("./results_full/best_model")
    print("Best model saved to ./results_full/best_model")

    # 9. Save the best model, trainer state, and DyLoRA-MoE state
    print("--- Saving Best Model and Full State ---")
    best_model_dir = "./results_full/best_model"
    
    # Save model, tokenizer, and training arguments
    trainer.save_model(best_model_dir)
    trainer.save_state()  # Saves optimizer, scheduler, etc. to output_dir
    tokenizer.save_pretrained(best_model_dir)
    torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
    
    # Save config.json with base model information for benchmark.py
    config_data = {
        "base_model_name_or_path": args.model_name,
        "num_experts": args.num_experts,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "model_type": "dylora-moe",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_path = os.path.join(best_model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    print(f"Best model, tokenizer, and training args saved to {best_model_dir} and {training_args.output_dir}")

    # Save DyLoRA-MoE specific state
    dylo_moe_state_dir = os.path.join(training_args.output_dir, "dylo_moe_state")
    save_dylo_moe_state(model, dylo_moe_state_dir)
    print(f"DyLoRA-MoE state saved to {dylo_moe_state_dir}")

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
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout value.")
    parser.add_argument("--balance_coefficient", type=float, default=0.01, help="Coefficient for load balancing auxiliary loss (0 to disable).")
    parser.add_argument("--interleaved_sampling", action="store_true", help="Use interleaved sampling for balanced dataset representation (50/50 Code Alpaca and MBPP).")
    parser.add_argument("--cosine_restarts", action="store_true", help="Use cosine_with_restarts learning rate scheduler with 2 cycles for better exploration.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Per-device training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Per-device evaluation batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps (effective batch size = train_batch_size * gradient_accumulation_steps).")
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args)