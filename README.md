# DyLoRA-MoE: Dynamic LoRA-based Mixture-of-Experts Architecture

A parameter-efficient Mixture-of-Experts architecture using LoRA (Low-Rank Adaptation) for training multiple specialized experts that share a frozen base model.

## Overview

DyLoRA-MoE combines the efficiency of LoRA with the specialization power of Mixture-of-Experts (MoE). The architecture features:

- **Shared Frozen Base Model**: All experts share the same frozen foundation model weights (~2.5B params)
- **Multiple LoRA Experts**: Lightweight adapter modules (~16M params each) that specialize on different tasks
- **Dynamic Router**: Learns to route inputs to the most appropriate expert(s)
- **Memory Efficient**: Only ~2.5% memory overhead per additional expert

### Key Features

- ✅ **Weight Sharing**: All experts share frozen base model weights (automatic via PEFT)
- ✅ **Parameter Efficient**: Only LoRA adapters and router are trainable (~2-3% of total params)
- ✅ **Scalable**: Add more experts with minimal memory cost
- ✅ **Flexible**: Supports both joint training and continual learning modes

For more details, refer to the [Technical Paper](DyLoRA%20-%20Technical%20Paper.md) and [Technical Design Document](DyLoRA-TDD.md).

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Hugging Face account with API token

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DyLoRA
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Hugging Face token:**
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```

### Training

#### Quick Start (Local Training)

Train with default settings (4 experts, joint training on Code Alpaca + MBPP):

```bash
python train.py --bf16 --num_epochs 10
```

#### Custom Configuration

```bash
# Use 8 experts with custom LoRA rank
python train.py \
  --num_experts 8 \
  --lora_r 32 \
  --lora_alpha 64 \
  --bf16 \
  --num_epochs 10

# Use custom datasets (80k Evol-Instruct + MBPP)
python train.py \
  --datasets "evol_instruct,mbpp" \
  --bf16 \
  --num_epochs 10

# Large-scale training with multiple datasets (~180k examples)
python train.py \
  --datasets "code_alpaca,evol_instruct,code_feedback,mbpp" \
  --bf16 \
  --num_epochs 10 \
  --balance_coefficient 0.01

# Quick test with data subset
python train.py \
  --training_subset 10 \
  --eval_subset 20 \
  --bf16 \
  --num_epochs 3
```

#### Available Datasets

See [DATASETS.md](DATASETS.md) for complete dataset documentation.

Available datasets (use with `--datasets "dataset1,dataset2,..."`):
- `code_alpaca`: 20k multilingual code instructions (default)
- `mbpp`: 1k Python programming problems (default)
- `evol_instruct`: 80k high-quality evolved code instructions ⭐
- `code_feedback`: Filtered code instructions with multi-language support ⭐
- `python_codes_25k`: 25k Python code examples
- `python_code_instructions_18k`: 18k Python instructions (Alpaca format)
- `python_code_23k_sharegpt`: 23k Python conversations (ChatGPT style)

#### Available Arguments

- `--num_experts`: Number of LoRA experts (default: 4)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling (default: 32)
- `--lora_dropout`: LoRA dropout rate (default: 0.05)
- `--num_epochs`: Training epochs (default: 10)
- `--training_subset`: Percentage of training data to use
- `--eval_subset`: Percentage of eval data to use
- `--bf16`: Use BF16 mixed precision (recommended)
- `--fp16`: Use FP16 mixed precision
- `--resume_from_checkpoint`: Resume from latest checkpoint (legacy - local files)
- `--wandb_checkpoint_artifact`: Resume from W&B artifact (e.g., "username/project/artifact:v0")
- `--datasets`: Comma-separated list of datasets (default: "code_alpaca,mbpp")
- `--interleaved_sampling`: Use 50/50 balanced sampling (works with 2 datasets)
- `--balance_coefficient`: Load balancing loss coefficient (default: 0.01)

### Checkpoint Resumption

DyLoRA-MoE supports resuming training from Weights & Biases checkpoints:

```bash
# Resume from W&B artifact (works everywhere - local & cloud)
python train.py \
  --wandb_checkpoint_artifact "johnrcumming/dylo-moe-full-training/best-dylora-model-full:v0" \
  --datasets "code_alpaca,mbpp" \
  --bf16 \
  --num_epochs 10

# For Vertex AI: Edit submit_full_training.py
# Set: WANDB_CHECKPOINT_ARTIFACT = "username/project/artifact:version"
```

See [CHECKPOINT_RESUMPTION.md](CHECKPOINT_RESUMPTION.md) for complete guide with examples.

### Docker Setup

DyLoRA-MoE images are available on Docker Hub for easy local development:

```bash
# Pull latest image
docker pull johnrcumming/dylora-moe:latest

# Run training locally in Docker
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  johnrcumming/dylora-moe:latest \
  python train.py --bf16 --num_epochs 3 --training_subset 10
```

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for complete Docker Hub guide.

### Vertex AI Training

For cloud-based training on Google Cloud Vertex AI:

```bash
# Build and push Docker image (to GCP Artifact Registry + Docker Hub)
./build_and_push.sh

# Submit training job
python submit_full_training.py
```

See [VERTEX_TRAINING.md](VERTEX_TRAINING.md) for detailed cloud training instructions.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                   DyLoRA-MoE Model                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Frozen Base Model (CodeGemma-2B)                    │  │
│  │  ~2.5B parameters (shared by all experts)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Dynamic Router                                       │  │
│  │  Learns which expert(s) to use per token/sequence    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │Expert 0 │  │Expert 1 │  │Expert 2 │  │Expert 3 │      │
│  │ LoRA    │  │ LoRA    │  │ LoRA    │  │ LoRA    │      │
│  │ ~16M    │  │ ~16M    │  │ ~16M    │  │ ~16M    │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Weighted Combination                                 │  │
│  │  Combine expert outputs using router weights         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Memory Efficiency

Example with 4 experts (r=16):

| Component | Parameters | Memory (BF16) | Trainable |
|-----------|------------|---------------|-----------|
| Base Model | 2.5B | ~5GB | ❌ Frozen |
| Expert 0 LoRA | 16M | ~32MB | ✅ |
| Expert 1 LoRA | 16M | ~32MB | ✅ |
| Expert 2 LoRA | 16M | ~32MB | ✅ |
| Expert 3 LoRA | 16M | ~32MB | ✅ |
| Router | 10K | ~20KB | ✅ |
| LM Head | 256M | ~512MB | ❌ Frozen |
| **Total** | **2.82B** | **~5.64GB** | **2.37%** |

**Key Insight**: Adding more experts only increases memory by ~32MB each!

## Project Structure

```
.
├── dylo_moe/               # Core implementation
│   ├── model.py            # DyLoRA_MoE model
│   ├── router.py           # Dynamic router with sparse delegation
│   ├── expert.py           # Expert manager (LoRA adapters)
│   ├── novelty_detector.py # Skill novelty detection
│   ├── skill_library.py    # Skill embedding storage
│   ├── scheduler.py        # Learning rate scheduler
│   └── utils.py            # Utility functions
├── data/
│   └── prepare_data.py     # Dataset loading (Code Alpaca, MBPP)
├── train.py                # Main training script
├── submit_full_training.py # Vertex AI job submission
├── build_and_push.sh       # Docker build for cloud training
└── requirements.txt        # Dependencies
```

## Training Modes

### Joint Training (Current Default)

All experts train together on the combined dataset, allowing the router to learn natural specialization patterns.

- **Datasets**: Code Alpaca + MBPP (merged)
- **Experts**: All created upfront and marked as mature
- **Router**: Uses sparse top-k delegation from the start
- **Benefits**: Stable training, natural specialization, better convergence

### Continual Learning (Optional)

Experts can be added dynamically as new skills are encountered.

- Enable by setting `allow_expert_growth=True` in model initialization
- Requires restructuring the training loop for sequential skill acquisition
- Useful for lifelong learning scenarios

## Implementation Details

### Weight Sharing

All LoRA experts automatically share the same frozen base model weights through PEFT's adapter mechanism:

```python
# First expert wraps the base model
model = get_peft_model(base_model, lora_config, adapter_name="expert_0")

# Additional experts share the base weights
model.add_adapter("expert_1", lora_config)  # Shares base_model weights
model.add_adapter("expert_2", lora_config)  # Shares base_model weights
```

Only the small LoRA A/B matrices (~16M params each) are unique per expert.

### Router Behavior

- **Sparse Delegation**: Uses top-k routing (default k=1) when all experts are mature
- **Dense Collaboration**: Falls back to softmax over all experts if new experts exist
- **Learnable**: Router weights are trained alongside LoRA adapters

### Frozen Components

- **Base Model**: All transformer layers frozen
- **LM Head**: Frozen (standard LoRA practice)
- **Embeddings**: Frozen

### Trainable Components

- **LoRA Adapters**: Low-rank matrices for each expert
- **Router**: Gating network that computes expert weights

## Monitoring

Training metrics logged to Weights & Biases:

- Training loss
- Per-domain evaluation losses (Code Alpaca, MBPP)
- Expert utilization (coming soon)
- Router entropy (coming soon)

## Performance Tips

1. **Start with 4-8 experts**: More experts don't always help
2. **Use BF16**: Better numerical stability than FP16
3. **Tune LoRA rank**: Higher rank (32-64) for complex tasks
4. **Monitor per-domain losses**: Ensure both datasets improve
5. **Check expert utilization**: Avoid dead experts

## Citation

If you use this code, please cite:

```bibtex
@misc{dylora-moe,
  title={DyLoRA-MoE: Dynamic LoRA-based Mixture-of-Experts},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

[Your License Here]

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Mixture of Experts](https://arxiv.org/abs/1701.06538)
