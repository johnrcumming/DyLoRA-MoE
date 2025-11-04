# DyLoRA-MoE AI Coding Assistant Instructions# DyLoRA-MoE AI Coding Assistant Instructions



## Project Overview## Project Overview

DyLoRA-MoE is a **parameter-efficient Mixture-of-Experts architecture** using LoRA adapters for training specialized experts that share a frozen base model (CodeGemma-2B). Each expert adds ~2.5% memory overhead (~16M params) while the frozen base remains at ~2.5B params.DyLoRA-MoE is a **parameter-efficient Mixture-of-Experts architecture** using LoRA (Low-Rank Adaptation) for training multiple specialized experts that share a frozen base model (CodeGemma-2B). Each expert adds only ~2.5% memory overhead (~16M params) while the frozen base remains at ~2.5B params.



## Core Architecture (Three Components)## Core Architecture Pattern



### 1. ExpertManager (`dylo_moe/expert.py`)### Weight Sharing via PEFT

Manages LoRA adapter lifecycle. **Critical**: All experts share the same frozen base model weights via PEFT:All experts **automatically share the same frozen base model weights** via the PEFT library. When you see code like:

```python```python

self.model = get_peft_model(self.model, peft_config, adapter_name="expert_0")self.model = get_peft_model(self.model, peft_config, adapter_name="expert_0")

self.model.add_adapter("expert_1", peft_config)  # Shares base weights, NOT a copy!self.model.add_adapter("expert_1", peft_config)  # Shares base weights!

``````

- `set_active_expert(i)`: Switches active PEFT adapter (single-expert training)This is NOT duplicating the base model—only the tiny LoRA adapter matrices (A and B) are unique per expert.

- `activate_all_experts()`: Enables multi-expert MoE routing (sets all adapters active via `base_model.set_adapter([list])`)

- Target modules auto-detected: `["q_proj", "v_proj", "k_proj", "o_proj", "query_key_value", "c_attn", "Wqkv"]`### Three-Component System

1. **ExpertManager** (`dylo_moe/expert.py`): Manages LoRA adapter lifecycle via PEFT's `add_adapter()` and `set_adapter()`

### 2. DynamicHybridRouter (`dylo_moe/router.py`)2. **DynamicHybridRouter** (`dylo_moe/router.py`): Learns to route inputs to experts; switches between dense (training) and sparse (inference) modes based on `expert_maturity` state

Learns to route inputs to experts. **Switches modes based on training state**:3. **DyLoRA_MoE** (`dylo_moe/model.py`): Orchestrates forward passes by iterating through experts and combining outputs weighted by router

- **Dense (training)**: Softmax over all experts when `model.training == True` or `expert_maturity == 0` (ensures gradients flow)

- **Sparse (inference)**: Top-k selection when all experts mature (efficient inference)### Critical Forward Pass Pattern

- **Device alignment required**: Always check `self.expert_maturity.device != logits.device` and move if neededMulti-expert forward passes iterate through experts but maintain a **single computational graph**:

```python

### 3. DyLoRA_MoE (`dylo_moe/model.py`)for i in range(self.router.num_experts):

Orchestrates forward passes with **single-pass MoE routing** (NOT N+1 passes):    self.expert_manager.set_active_expert(i)  # Switches PEFT adapter, not model

```python    expert_out = self.foundation_model(input_ids, ...)

# Step 1: Get hidden states    weighted_logits = expert_out.logits * expert_weights[:, i].unsqueeze(1).unsqueeze(2)

base_outputs = self.foundation_model(input_ids, output_hidden_states=True)    logits = logits + weighted_logits if logits else weighted_logits

hidden_states = base_outputs.hidden_states[-1]```

**Never** create separate model instances per expert—this defeats the memory efficiency design.

# Step 2: Compute routing weights (with gradients!)

routing_weights = self.router(hidden_states)  # [batch, seq, num_experts]## Training Workflows



# Step 3: Activate all experts and single forward pass with routing### Local Training (Development)

self.expert_manager.activate_all_experts()```bash

outputs = self.foundation_model(input_ids, routing_weights=routing_weights)python train.py --bf16 --num_epochs 10 --num_experts 4

``````

**Never** create separate model instances per expert or loop through experts with multiple forward passes.- Uses Hugging Face `Trainer` with custom callbacks for gradient/routing monitoring

- Datasets: Code Alpaca (multilingual) + MBPP (Python coding challenges)

## Training Workflows- Automatic validation splits: 90/10 for Code Alpaca, 70/10/20 train/val/test for MBPP



### Local Development### Cloud Training (Vertex AI)

```bash```bash

python train.py --bf16 --num_epochs 10 --num_experts 4./build_and_push.sh  # Builds Docker image, pushes to Artifact Registry

python train.py --training_subset 10 --eval_subset 20 --num_epochs 3  # Quick testpython submit_full_training.py  # Submits CustomJob with spot VMs

``````

- Datasets: Code Alpaca (20k multilingual) + MBPP (1k Python) via `data/prepare_data.py`- Configured for `a3-highgpu-1g` (NVIDIA H100) or `a2-highgpu-1g` (A100)

- Dataset registry: `AVAILABLE_DATASETS` dict in `prepare_data.py` - use `get_dataset("name")`- **Critical**: Set `restart_job_on_worker_restart=True` for spot VM resilience

- Validation splits: Code Alpaca 90/10, MBPP 70/10/20 (train/val/test)- Checkpointing enabled via `--resume_from_checkpoint` flag



### Cloud Training (Vertex AI)### Key Training Flags

```bash- `--interleaved_sampling`: 50/50 balanced sampling from Code Alpaca/MBPP (prevents dataset imbalance)

./build_and_push.sh  # Builds/pushes to Artifact Registry + Docker Hub- `--balance_coefficient 0.01`: Adds load balancing auxiliary loss to prevent expert collapse

python submit_full_training.py  # Submits CustomJob- `--cosine_restarts`: LR scheduler with 2 warmup/decay cycles for better exploration

```- `--train_batch_size 2 --gradient_accumulation_steps 32`: Small batch + accumulation for multi-expert memory constraints

- Machine types: `a3-highgpu-1g` (H100) or `a2-highgpu-1g` (A100)

- Spot VM resilience: `restart_job_on_worker_restart=True` in job config## Development Patterns

- Checkpoint resumption via `--wandb_checkpoint_artifact "user/project/artifact:v0"`

### Module Discovery for LoRA

### Critical Training FlagsTarget modules are **auto-detected** in `ExpertManager.create_expert()`:

- `--interleaved_sampling`: 50/50 balanced sampling (prevents MBPP underfitting due to size mismatch)```python

- `--balance_coefficient 0.01`: Load balancing auxiliary loss (prevents expert collapse)possible_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "query_key_value", "c_attn", "Wqkv"]

- `--train_batch_size 2 --gradient_accumulation_steps 32`: Memory-efficient training (do NOT increase batch size)```

- `--datasets "code_alpaca,mbpp"`: Comma-separated dataset names (default)If adding a new base model, ensure its attention layers match these patterns or extend the list.



### Advanced Datasets### Gradient Monitoring

```bashTwo custom `TrainerCallback` classes in `train.py`:

# Large-scale training (~180k examples)- `GradientMonitoringCallback`: Logs per-expert gradient norms to wandb (extracts expert ID from param name like `.expert_0.weight`)

python train.py --datasets "code_alpaca,evol_instruct,code_feedback,mbpp" --bf16- `DyLoRAMonitoringCallback`: Logs routing weights, expert usage distribution, load balancing metrics

```

Available datasets: `code_alpaca`, `mbpp`, `evol_instruct` (80k), `code_feedback`, `python_codes_25k`, `python_code_instructions_18k`, `python_code_23k_sharegpt`### Device Management

Router and model **must** be on the same device. Pattern used throughout:

## Benchmarking & Evaluation```python

if self.expert_maturity.device != logits.device:

### EvalPlus Framework (Default)    self.expert_maturity = self.expert_maturity.to(logits.device)

```bash```

python benchmark.py --benchmarks humaneval  # Default: EvalPlus for standardized eval

python benchmark.py --trained_model ./results/best_model --max_samples 164### Mixed Precision Compatibility

python benchmark.py --wandb_artifact "user/project/model:v0" --max_samples 50Router uses explicit dtype matching for scatter operations:

``````python

- Uses EvalPlus for accurate, comparable results (aligns with published benchmarks)routing_weights = routing_weights.scatter(-1, top_k_indices, 

- Supports HumanEval, HumanEval+, MBPP, MBPP+                                         F.softmax(top_k_weights, dim=-1).to(routing_weights.dtype))

- `--no_evalplus`: Falls back to legacy custom benchmarks (not recommended)```

Always use `.to(dtype)` when combining tensors in routing logic to avoid BF16/FP32 mismatches.

### Docker Entrypoint (Unified Interface)

```bash## Testing & Validation

# Training

docker run --gpus all -e HF_TOKEN=$HF_TOKEN johnrcumming/dylora-moe:latest \### Quick Validation

  --train --datasets "code_alpaca,mbpp" --num_epochs 3```bash

python train.py --training_subset 10 --eval_subset 20 --num_epochs 3 --bf16

# Benchmarking```

docker run --gpus all -e HF_TOKEN=$HF_TOKEN johnrcumming/dylora-moe:latest \Uses 10% of training data, 20% of eval data for fast iteration.

  --benchmark --base-model --max-samples 20

```### Common Issues

- `entrypoint.py`: Unified script for training/benchmarking in containers- **OOM errors**: Reduce `--train_batch_size` (default 2 is already low) or increase `--gradient_accumulation_steps`

- Platform detection: Auto-detects Vertex AI vs Vast.ai for optimizations- **Router not learning**: Check `balance_coefficient > 0` and verify dense routing is active during training (`expert_maturity == 0` or `model.training == True`)

- **Expert collapse**: All inputs route to one expert—increase `--balance_coefficient` or use `--interleaved_sampling`

## Development Patterns

## File Organization

### Custom TrainerCallback Pattern- `dylo_moe/`: Core model components (avoid adding training logic here—use callbacks in `train.py`)

Both callbacks in `train.py` use `TrainerCallback` interface:- `data/prepare_data.py`: Dataset loaders with validation split logic

```python- `train.py`: Main training script with `argparse` CLI (780+ lines—refactor cautiously)

class GradientMonitoringCallback(TrainerCallback):- `submit_full_training.py`: Vertex AI job submission (requires GCP credentials in `dylora-ece7d9f1337d.json`)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):

        # Extract expert ID from param names like "...lora_A.expert_0.weight"## Environment Setup

        for name, param in self.model.named_parameters():```bash

            if param.grad is not None and ".expert_" in name:python3 -m venv .venv && source .venv/bin/activate

                expert_id = extract_expert_id(name)  # Parse ".expert_N"pip install -r requirements.txt

```export HF_TOKEN=your_token  # Required for CodeGemma model access

- `GradientMonitoringCallback`: Logs per-expert gradient norms (matches `.expert_N` in param names)export WANDB_API_KEY=your_key  # Optional but recommended for monitoring

- `DyLoRAMonitoringCallback`: Tracks routing stats, expert usage, load balancing metrics```



### Mixed Precision Device Management## Key Design Decisions

**Always explicitly match dtypes** to avoid BF16/FP32 errors:- **Why freeze base model**: Only train LoRA adapters + router (~2-3% of params) for efficiency

```python- **Why dynamic routing**: Dense routing (softmax) during training for gradient flow; sparse (top-k) at inference for efficiency

# Router scatter operation- **Why balance loss**: Prevents router from collapsing to single expert (common MoE failure mode)

routing_weights = routing_weights.scatter(- **Why interleaved sampling**: MBPP is much smaller than Code Alpaca—without balancing, model overfits to Code Alpaca patterns

    -1, top_k_indices, 
    F.softmax(top_k_weights, dim=-1).to(routing_weights.dtype)  # Explicit dtype match
)

# Device alignment (router must match model device)
if self.expert_maturity.device != logits.device:
    self.expert_maturity = self.expert_maturity.to(logits.device)
```

### Testing Patterns
Tests use standalone functions (no unittest framework):
```python
def test_routing_weights_basic():
    """Test routing_weights parameter works correctly"""
    model = get_peft_model(base_model, lora_config, adapter_name="expert_0")
    model.add_adapter("expert_1", lora_config)
    model.base_model.set_adapter(["expert_0", "expert_1"])  # Multi-expert mode
    outputs = model(input_ids, routing_weights=routing_weights)
    assert outputs.logits.shape == expected_shape
```
Run directly: `python test_routing_weights.py` (no pytest required)

## Common Issues & Solutions

### OOM Errors
- Reduce `--train_batch_size` (default 2 is already minimal)
- Increase `--gradient_accumulation_steps` (default 32)
- Use `--bf16` instead of `--fp16` (better stability)

### Router Not Learning
- Check `balance_coefficient > 0` (default 0.01)
- Verify dense routing active: `model.training == True` or `expert_maturity == 0`
- Monitor wandb logs: `grad_norm/router` should be non-zero

### Expert Collapse (All inputs route to one expert)
- Increase `--balance_coefficient` (try 0.05 or 0.1)
- Enable `--interleaved_sampling` for balanced dataset sampling
- Check expert usage in wandb: `expert_usage/expert_N` should be similar across experts

### Checkpoint Resumption
- W&B artifacts (recommended): `--wandb_checkpoint_artifact "user/project/artifact:v0"`
- Local files (legacy): `--resume_from_checkpoint` (auto-finds latest)
- Config preserved: LoRA rank, num_experts, router state all restored

## File Structure & Boundaries
```
dylo_moe/          # Core model logic ONLY (no training code)
├── model.py       # DyLoRA_MoE orchestration
├── expert.py      # ExpertManager (PEFT adapters)
├── router.py      # DynamicHybridRouter
└── utils.py       # Helper functions

data/prepare_data.py   # Dataset registry & loaders
train.py               # Training script (780 lines - callbacks, argparse CLI)
benchmark.py           # Evaluation suite (EvalPlus + legacy)
entrypoint.py          # Docker unified interface
submit_full_training.py  # Vertex AI job submission
```
**Rule**: Training logic belongs in `train.py` callbacks, NOT in `dylo_moe/`.

## Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
set HF_TOKEN=your_token        # Windows PowerShell (use $env:HF_TOKEN for PS Core)
set WANDB_API_KEY=your_key     # Optional but recommended
```

## Design Rationale
- **Frozen base model**: Train only LoRA adapters + router (~2-3% params) for efficiency
- **Dense routing (training)**: Softmax ensures gradients flow to all experts and router learns effectively
- **Sparse routing (inference)**: Top-k reduces compute cost once experts specialize
- **Load balancing loss**: Prevents router collapse (common MoE failure mode where one expert dominates)
- **Interleaved sampling**: MBPP is 20x smaller than Code Alpaca - balanced sampling prevents overfitting to larger dataset
