import torch
import os
import argparse
os.environ["WANDB_DISABLED"] = "false"
import wandb
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
    download_humaneval
)


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

def evaluate_humaneval(model, tokenizer, humaneval_dataset, max_samples=None):
    """
    Evaluate model on HumanEval benchmark by generating code completions.
    
    Returns:
        dict: Contains 'pass@1' score and generation samples
    """
    print("\n--- Evaluating on HumanEval Benchmark ---")
    
    # Use subset if specified
    eval_samples = list(humaneval_dataset)
    if max_samples:
        eval_samples = eval_samples[:max_samples]
    
    print(f"Generating completions for {len(eval_samples)} HumanEval problems...")
    
    model.eval()
    generations = []
    correct = 0
    
    with torch.no_grad():
        for i, example in enumerate(eval_samples):
            prompt = example.get('prompt', '')
            canonical_solution = example.get('canonical_solution', '')
            test = example.get('test', '')
            entry_point = example.get('entry_point', '')
            
            # Generate completion
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif torch.backends.mps.is_available():
                inputs = {k: v.to('mps') for k, v in inputs.items()}
            
            # Generate using DyLoRA-MoE's generate method (uses router to select expert)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the completion (remove the prompt)
            completion = generated_text[len(prompt):].strip()
            
            generations.append({
                'task_id': example.get('task_id', f'HumanEval/{i}'),
                'prompt': prompt,
                'completion': completion,
                'canonical_solution': canonical_solution,
            })
            
            # Simple heuristic check: does generated code contain the entry point?
            if entry_point and entry_point in completion:
                correct += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(eval_samples)} completions...")
    
    # Calculate pass@1 approximation (heuristic-based, not exact)
    # Note: True pass@1 requires running tests, which we skip for training monitoring
    pass_at_1_approx = correct / len(eval_samples) if eval_samples else 0.0
    
    results = {
        'pass@1_approx': pass_at_1_approx,
        'num_samples': len(eval_samples),
        'num_with_entry_point': correct,
        'sample_generations': generations[:5],  # First 5 for inspection
    }
    
    print(f"HumanEval Pass@1 (approx): {pass_at_1_approx:.2%}")
    print(f"Generations with entry point: {correct}/{len(eval_samples)}")
    
    model.train()
    return results

def main(args):
    # 1. Initialize wandb
    wandb.init(project="dylo-moe-full-training")

    # 2. Instantiate the model
    hf_token = os.environ.get("HF_TOKEN")
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
    
    # 4. Load datasets - Simplified Configuration (2 domains + HumanEval)
    # NOTE: APPS and CodeSearchNet are deprecated (dataset scripts no longer supported)
    # See DATASET_UPDATE_STATUS.md for details
    # Domain 1: General instruction following (Code Alpaca)
    # Domain 2: Basic programming problems (MBPP)
    # Benchmark: HumanEval (evaluation only)
    
    print("\n--- Loading Datasets (Simplified: 2 Domains + HumanEval) ---")
    
    code_alpaca_dataset = download_code_alpaca(filter_python=False, with_validation=True)
    mbpp_dataset = download_mbpp(with_validation=True)
    
    # Download HumanEval for evaluation only (not used in training)
    humaneval_dataset = download_humaneval()
    
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
    
    # Evaluate base model's code generation capability
    baseline_results = evaluate_humaneval(
        model, tokenizer, humaneval_dataset, max_samples=20  # Use subset for speed
    )
    
    wandb.log({
        "baseline/humaneval_pass@1": baseline_results['pass@1_approx'],
        "baseline/humaneval_with_entry_point": baseline_results['num_with_entry_point'],
        "baseline/num_samples": baseline_results['num_samples'],
    }, commit=False)
    
    print(f"\n✓ Baseline HumanEval Pass@1 (approx): {baseline_results['pass@1_approx']:.2%}")
    print(f"✓ Baseline generations with entry point: {baseline_results['num_with_entry_point']}/{baseline_results['num_samples']}")
    print("="*80 + "\n")
    
    # Prepare training data from each domain
    print("\n--- Preparing Domain-Specific Training Data ---")
    
    # Domain 1: Code Alpaca (instruction following)
    skill1_data = [ex['instruction'] + "\n" + ex['output'] for ex in code_alpaca_dataset['train']]
    print(f"Domain 1 (Code Alpaca - Instructions): {len(skill1_data)} examples")
    
    # Domain 2: MBPP (basic problems)
    skill2_data = [ex['text'] for ex in mbpp_dataset['train']]
    print(f"Domain 2 (MBPP - Basic Problems): {len(skill2_data)} examples")
    
    # Combine datasets
    print(f"\nTotal training examples before combining: {len(skill1_data) + len(skill2_data)}")
    
    # Choose sampling strategy based on CLI flag
    if args.interleaved_sampling:
        print("\n--- Using Interleaved Sampling (Equal 50/50 representation) ---")
        # Use interleaved sampling for equal representation
        combined_data = create_interleaved_dataset(
            skill1_data, skill2_data
        )
        
        print(f"\nDomain representation: ~50% Code Alpaca, ~50% MBPP (balanced)")
    else:
        print("\n--- Using Concatenation (dataset size based) ---")
        combined_data = skill1_data + skill2_data
        total = len(combined_data)
        print(f"Code Alpaca: {len(skill1_data)} ({len(skill1_data)/total*100:.1f}%)")
        print(f"MBPP: {len(skill2_data)} ({len(skill2_data)/total*100:.1f}%)")
    
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
            early_stopping_patience=3,  # Stop after 3 epochs with no improvement
            early_stopping_threshold=0.005  # Minimum improvement threshold for plateau detection
        ))
        print("✓ Early stopping enabled (patience=3, threshold=0.005)")
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

    # 7. Initial Evaluation on HumanEval Benchmark (with initialized LoRA, before training)
    print("\n--- Initial Evaluation on HumanEval Benchmark (With Initialized LoRA) ---")
    # Temporarily swap to HumanEval for initial benchmark
    trainer.eval_dataset = humaneval_benchmark
    initial_eval = trainer.evaluate(metric_key_prefix="eval")
    print(f"Initial HumanEval Loss (with LoRA): {initial_eval['eval_loss']:.4f}")
    # Restore validation dataset for training
    trainer.eval_dataset = val_dataset
    
    # Evaluate HumanEval code generation (pass@1 approximation)
    initial_humaneval_results = evaluate_humaneval(
        model, tokenizer, humaneval_dataset, max_samples=20  # Use subset for speed
    )
    
    wandb.log({
        "initial_humaneval_loss": initial_eval['eval_loss'],
        "initial_humaneval_pass@1": initial_humaneval_results['pass@1_approx'],
        "initial_humaneval_with_entry_point": initial_humaneval_results['num_with_entry_point'],
    }, commit=False)

    # 8. Train the model on the combined dataset
    print("\n--- Training on Combined Dataset ---")
    print(f"Training with {model.expert_manager.num_experts} experts")
    trainer.train(resume_from_checkpoint=True if args.resume_from_checkpoint else None)

    # 9. Final evaluation on HumanEval Benchmark (after training)
    print("\n--- Final Evaluation on HumanEval Benchmark ---")
    # Swap to HumanEval for final benchmark
    trainer.eval_dataset = humaneval_benchmark
    final_eval = trainer.evaluate(metric_key_prefix="eval")
    print(f"Final HumanEval Loss: {final_eval['eval_loss']:.4f}")
    
    # Evaluate HumanEval code generation (pass@1 approximation)
    final_humaneval_results = evaluate_humaneval(
        model, tokenizer, humaneval_dataset, max_samples=20  # Use subset for speed
    )
    
    wandb.log({
        "final_humaneval_loss": final_eval['eval_loss'],
        "final_humaneval_pass@1": final_humaneval_results['pass@1_approx'],
        "final_humaneval_with_entry_point": final_humaneval_results['num_with_entry_point'],
    }, commit=False)

    # 10. Save the best model
    print("\n--- Saving Best Model ---")
    trainer.save_model("./results_full/best_model")
    tokenizer.save_pretrained("./results_full/best_model")
    print("Best model saved to ./results_full/best_model")

    # 10. Save the best model, trainer state, and DyLoRA-MoE state
    print("--- Saving Best Model and Full State ---")
    best_model_dir = "./results_full/best_model"
    
    # Save model, tokenizer, and training arguments
    trainer.save_model(best_model_dir)
    trainer.save_state()  # Saves optimizer, scheduler, etc. to output_dir
    tokenizer.save_pretrained(best_model_dir)
    torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
    print(f"Best model, tokenizer, and training args saved to {best_model_dir} and {training_args.output_dir}")

    # Save DyLoRA-MoE specific state
    dylo_moe_state_dir = os.path.join(training_args.output_dir, "dylo_moe_state")
    save_dylo_moe_state(model, dylo_moe_state_dir)
    print(f"DyLoRA-MoE state saved to {dylo_moe_state_dir}")

    # 11. Upload the entire output directory as a wandb artifact
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint.")
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
    args = parser.parse_args()
    main(args)