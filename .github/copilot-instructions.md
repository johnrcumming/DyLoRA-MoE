# DyLoRA-MoE AI Coding Assistant Instructions

## Project Overview
DyLoRA-MoE is a **parameter-efficient Mixture-of-Experts architecture** using LoRA (Low-Rank Adaptation) for training multiple specialized experts that share a frozen base model (CodeGemma-2B). Each expert adds only ~2.5% memory overhead (~16M params) while the frozen base remains at ~2.5B params.

## Core Architecture Pattern

### Weight Sharing via PEFT
All experts **automatically share the same frozen base model weights** via the PEFT library. When you see code like:
```python
self.model = get_peft_model(self.model, peft_config, adapter_name="expert_0")
self.model.add_adapter("expert_1", peft_config)  # Shares base weights!
```
This is NOT duplicating the base model—only the tiny LoRA adapter matrices (A and B) are unique per expert.

### Three-Component System
1. **ExpertManager** (`dylo_moe/expert.py`): Manages LoRA adapter lifecycle via PEFT's `add_adapter()` and `set_adapter()`
2. **DynamicHybridRouter** (`dylo_moe/router.py`): Learns to route inputs to experts; switches between dense (training) and sparse (inference) modes based on `expert_maturity` state
3. **DyLoRA_MoE** (`dylo_moe/model.py`): Orchestrates forward passes by iterating through experts and combining outputs weighted by router

### Critical Forward Pass Pattern
Multi-expert forward passes iterate through experts but maintain a **single computational graph**:
```python
for i in range(self.router.num_experts):
    self.expert_manager.set_active_expert(i)  # Switches PEFT adapter, not model
    expert_out = self.foundation_model(input_ids, ...)
    weighted_logits = expert_out.logits * expert_weights[:, i].unsqueeze(1).unsqueeze(2)
    logits = logits + weighted_logits if logits else weighted_logits
```
**Never** create separate model instances per expert—this defeats the memory efficiency design.

## Training Workflows

### Local Training (Development)
```bash
python train.py --bf16 --num_epochs 10 --num_experts 4
```
- Uses Hugging Face `Trainer` with custom callbacks for gradient/routing monitoring
- Datasets: Code Alpaca (multilingual) + MBPP (Python coding challenges)
- Automatic validation splits: 90/10 for Code Alpaca, 70/10/20 train/val/test for MBPP

### Cloud Training (Vertex AI)
```bash
./build_and_push.sh  # Builds Docker image, pushes to Artifact Registry
python submit_full_training.py  # Submits CustomJob with spot VMs
```
- Configured for `a3-highgpu-1g` (NVIDIA H100) or `a2-highgpu-1g` (A100)
- **Critical**: Set `restart_job_on_worker_restart=True` for spot VM resilience
- Checkpointing enabled via `--resume_from_checkpoint` flag

### Key Training Flags
- `--interleaved_sampling`: 50/50 balanced sampling from Code Alpaca/MBPP (prevents dataset imbalance)
- `--balance_coefficient 0.01`: Adds load balancing auxiliary loss to prevent expert collapse
- `--cosine_restarts`: LR scheduler with 2 warmup/decay cycles for better exploration
- `--train_batch_size 2 --gradient_accumulation_steps 32`: Small batch + accumulation for multi-expert memory constraints

## Development Patterns

### Module Discovery for LoRA
Target modules are **auto-detected** in `ExpertManager.create_expert()`:
```python
possible_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "query_key_value", "c_attn", "Wqkv"]
```
If adding a new base model, ensure its attention layers match these patterns or extend the list.

### Gradient Monitoring
Two custom `TrainerCallback` classes in `train.py`:
- `GradientMonitoringCallback`: Logs per-expert gradient norms to wandb (extracts expert ID from param name like `.expert_0.weight`)
- `DyLoRAMonitoringCallback`: Logs routing weights, expert usage distribution, load balancing metrics

### Device Management
Router and model **must** be on the same device. Pattern used throughout:
```python
if self.expert_maturity.device != logits.device:
    self.expert_maturity = self.expert_maturity.to(logits.device)
```

### Mixed Precision Compatibility
Router uses explicit dtype matching for scatter operations:
```python
routing_weights = routing_weights.scatter(-1, top_k_indices, 
                                         F.softmax(top_k_weights, dim=-1).to(routing_weights.dtype))
```
Always use `.to(dtype)` when combining tensors in routing logic to avoid BF16/FP32 mismatches.

## Testing & Validation

### Quick Validation
```bash
python train.py --training_subset 10 --eval_subset 20 --num_epochs 3 --bf16
```
Uses 10% of training data, 20% of eval data for fast iteration.

### Common Issues
- **OOM errors**: Reduce `--train_batch_size` (default 2 is already low) or increase `--gradient_accumulation_steps`
- **Router not learning**: Check `balance_coefficient > 0` and verify dense routing is active during training (`expert_maturity == 0` or `model.training == True`)
- **Expert collapse**: All inputs route to one expert—increase `--balance_coefficient` or use `--interleaved_sampling`

## File Organization
- `dylo_moe/`: Core model components (avoid adding training logic here—use callbacks in `train.py`)
- `data/prepare_data.py`: Dataset loaders with validation split logic
- `train.py`: Main training script with `argparse` CLI (780+ lines—refactor cautiously)
- `submit_full_training.py`: Vertex AI job submission (requires GCP credentials in `dylora-ece7d9f1337d.json`)

## Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=your_token  # Required for CodeGemma model access
export WANDB_API_KEY=your_key  # Optional but recommended for monitoring
```

## Key Design Decisions
- **Why freeze base model**: Only train LoRA adapters + router (~2-3% of params) for efficiency
- **Why dynamic routing**: Dense routing (softmax) during training for gradient flow; sparse (top-k) at inference for efficiency
- **Why balance loss**: Prevents router from collapsing to single expert (common MoE failure mode)
- **Why interleaved sampling**: MBPP is much smaller than Code Alpaca—without balancing, model overfits to Code Alpaca patterns
