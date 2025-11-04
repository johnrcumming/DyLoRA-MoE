# PeftMoEDecoder Implementation Guide (Phase 1)

> **Part of**: [EVALPLUS_PEFT_MOE_INTEGRATION_PLAN.md](./EVALPLUS_PEFT_MOE_INTEGRATION_PLAN.md)
> 
> **Status**: Implementation Ready
> 
> **Estimated Effort**: 3-4 days

## Overview

This guide provides step-by-step implementation details for creating the **PeftMoEDecoder** provider - the core component that extends EvalPlus to support PEFT models with MoE architectures.

## File Structure

```
evalplus/
└── evalplus/
    └── provider/
        ├── __init__.py          (modify - add peft_moe backend)
        ├── base.py              (no changes)
        ├── hf.py                (no changes)
        ├── peft_moe.py          (NEW - create this file)
        └── utility.py           (no changes)
```

## Implementation Steps

### Step 1: Create Base Structure

**File**: `evalplus/evalplus/provider/peft_moe.py`

```python
"""
PEFT MoE Decoder Provider for EvalPlus

Supports evaluation of PEFT (Parameter-Efficient Fine-Tuning) models with
Mixture-of-Experts (MoE) architectures, including:
- DyLoRA-MoE
- X-LoRA
- Standard PEFT adapters (LoRA, QLoRA, AdaLoRA)
- W&B artifacts
- Multiple routing strategies
"""

from typing import List, Optional, Dict, Any, Union
import os
import json
import torch
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)

from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class PeftMoEDecoder(DecoderBase):
    """
    EvalPlus provider for PEFT models with MoE support.
    
    Example usage:
        # DyLoRA-MoE with router
        decoder = PeftMoEDecoder(
            name="./trained_model",
            base_model="google/codegemma-2b",
            routing_strategy="router"
        )
        
        # Single expert
        decoder = PeftMoEDecoder(
            name="./trained_model",
            base_model="google/codegemma-2b",
            routing_strategy="single:0"
        )
        
        # W&B artifact
        decoder = PeftMoEDecoder(
            name="user/project/model:v0",
            wandb_artifact="user/project/model:v0",
            base_model="google/codegemma-2b"
        )
    """
    
    def __init__(
        self,
        name: str,
        dataset: str,
        base_model: Optional[str] = None,
        adapter_path: Optional[str] = None,
        routing_strategy: str = "router",
        wandb_artifact: Optional[str] = None,
        force_base_prompt: bool = False,
        attn_implementation: str = "eager",
        device_map: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize PEFT MoE decoder.
        
        Args:
            name: Model name/path or W&B artifact
            dataset: Dataset name (for EOS tokens)
            base_model: Base model name (e.g., "google/codegemma-2b")
            adapter_path: Explicit adapter path (optional)
            routing_strategy: Expert routing strategy:
                - "router": Use model's internal router (default)
                - "single:<id>": Use specific expert (e.g., "single:0")
                - "best": Analyze prompt, select best expert
                - "ensemble": Generate with all experts, combine
                - "round_robin": Cycle through experts
            wandb_artifact: W&B artifact path (e.g., "user/project/model:v0")
            force_base_prompt: Force base model prompts (not chat)
            attn_implementation: Attention implementation ("eager", "flash_attention_2", "sdpa")
            device_map: Device map for model loading
            **kwargs: Additional arguments for DecoderBase
        """
        super().__init__(name=name, **kwargs)
        
        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.routing_strategy = routing_strategy
        self.wandb_artifact = wandb_artifact
        self.force_base_prompt = force_base_prompt
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.dataset = dataset
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model format detection
        self.model_format = None
        self.model_path = name
        
        # Load model
        self._load_model()
        
        # Setup tokenizer
        self._setup_tokenizer()
        
        # Setup EOS tokens
        if self.is_direct_completion():
            self.eos += extra_eos_for_direct_completion(dataset)
        else:
            self.eos += ["\n```\n"]
        
        print(f"PeftMoEDecoder initialized:")
        print(f"  Model format: {self.model_format}")
        print(f"  Base model: {self.base_model_name}")
        print(f"  Routing strategy: {self.routing_strategy}")
        print(f"  Device: {self.device}")
        print(f"  EOS tokens: {self.eos}")
    
    def _load_model(self):
        """Load PEFT MoE model based on detected format."""
        # TODO: Implement in Step 2
        pass
    
    def _setup_tokenizer(self):
        """Setup tokenizer for the model."""
        # TODO: Implement in Step 3
        pass
    
    def is_direct_completion(self) -> bool:
        """Check if using direct completion (no chat template)."""
        return self.force_base_prompt or self.tokenizer.chat_template is None
    
    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        """
        Generate code completions using PEFT MoE model.
        
        Args:
            prompt: Input prompt
            do_sample: Use sampling (vs greedy)
            num_samples: Number of completions to generate
        
        Returns:
            List of generated code strings
        """
        # TODO: Implement in Step 4
        pass
```

### Step 2: Implement Model Loading

Add to `PeftMoEDecoder` class:

```python
def _load_model(self):
    """Load PEFT MoE model based on detected format."""
    
    # Step 1: Handle W&B artifact if provided
    if self.wandb_artifact:
        print(f"Downloading W&B artifact: {self.wandb_artifact}")
        self.model_path = self._load_from_wandb_artifact(self.wandb_artifact)
        print(f"Artifact downloaded to: {self.model_path}")
    
    # Step 2: Detect model format
    self.model_format = self._detect_model_format(self.model_path)
    print(f"Detected model format: {self.model_format}")
    
    # Step 3: Load base model if needed
    if self.base_model_name is None:
        self.base_model_name = self._infer_base_model(self.model_path)
        print(f"Inferred base model: {self.base_model_name}")
    
    # Step 4: Load model based on format
    if self.model_format == "dylora_moe":
        self.model = self._load_dylora_moe_model()
    elif self.model_format == "peft_adapter":
        self.model = self._load_peft_adapter_model()
    elif self.model_format == "merged":
        self.model = self._load_merged_model()
    else:
        raise ValueError(f"Unknown model format: {self.model_format}")
    
    # Step 5: Move to device if needed
    if self.device_map is None:
        self.model = self.model.to(self.device)
    
    print(f"✓ Model loaded successfully")


def _detect_model_format(self, model_path: str) -> str:
    """
    Detect PEFT model format.
    
    Returns:
        "dylora_moe": DyLoRA-MoE with router state
        "peft_adapter": Standard PEFT adapter
        "merged": Merged PEFT model
    """
    model_path = Path(model_path)
    
    # Check for DyLoRA-MoE indicators
    # 1. Check for dylo_moe_state directory (legacy)
    if (model_path.parent / "dylo_moe_state").exists():
        return "dylora_moe"
    
    # 2. Check config.json for DyLoRA-MoE marker
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if config.get("model_type") == "dylora-moe":
                return "dylora_moe"
            if config.get("_dylora_original_type") == "dylora-moe":
                return "dylora_moe"
    
    # 3. Check for PEFT adapter
    if (model_path / "adapter_config.json").exists():
        return "peft_adapter"
    
    # 4. Default to merged if has model weights
    if (model_path / "model.safetensors").exists() or (model_path / "pytorch_model.bin").exists():
        return "merged"
    
    raise ValueError(f"Cannot determine model format for: {model_path}")


def _infer_base_model(self, model_path: str) -> str:
    """
    Infer base model name from config.
    
    Returns:
        Base model name (e.g., "google/codegemma-2b")
    """
    config_path = Path(model_path) / "config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path")
            if base_model:
                return base_model
    
    # Check adapter_config.json for PEFT adapters
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            config = json.load(f)
            base_model = config.get("base_model_name_or_path")
            if base_model:
                return base_model
    
    # Default fallback
    default = "google/codegemma-2b"
    print(f"⚠️  Could not infer base model, using default: {default}")
    return default


def _load_from_wandb_artifact(self, artifact_path: str) -> str:
    """
    Download W&B artifact and return local path.
    
    Args:
        artifact_path: Format "entity/project/artifact:version"
    
    Returns:
        Local path to downloaded artifact
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb is required for artifact loading. "
            "Install with: pip install wandb"
        )
    
    # Initialize offline run for artifact download
    # Use offline mode to avoid creating unnecessary runs
    with wandb.init(mode="offline", anonymous="allow") as run:
        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
    
    # Look for model subdirectory
    artifact_path = Path(artifact_dir)
    
    # Check for best_model subdirectory (common in training artifacts)
    best_model_path = artifact_path / "best_model"
    if best_model_path.exists():
        return str(best_model_path)
    
    return str(artifact_path)


def _load_dylora_moe_model(self):
    """Load DyLoRA-MoE model."""
    print("Loading DyLoRA-MoE model...")
    
    # Try importing DyLoRA-MoE
    try:
        from dylo_moe.model import DyLoRA_MoE
        from dylo_moe.expert import ExpertManager
    except ImportError:
        raise ImportError(
            "DyLoRA-MoE modules not found. "
            "Ensure dylo_moe package is available."
        )
    
    # Load the model
    # DyLoRA-MoE models are typically saved as PEFT models with router state
    # We need to reconstruct the DyLoRA_MoE wrapper
    
    # Check if this is a merged model or PEFT structure
    config_path = Path(self.model_path) / "config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            num_experts = config.get("num_experts", 4)
            lora_r = config.get("lora_r", 16)
            lora_alpha = config.get("lora_alpha", 32)
    else:
        # Defaults
        num_experts = 4
        lora_r = 16
        lora_alpha = 32
    
    # Initialize DyLoRA_MoE model
    model = DyLoRA_MoE(
        model_name=self.base_model_name,
        num_experts=num_experts,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        token=os.environ.get("HF_TOKEN"),
    )
    
    # Load saved weights
    # Try safetensors first, then pytorch_model.bin
    weights_file = None
    safetensors_path = Path(self.model_path) / "model.safetensors"
    pytorch_path = Path(self.model_path) / "pytorch_model.bin"
    
    if safetensors_path.exists():
        weights_file = safetensors_path
        from safetensors.torch import load_file
        state_dict = load_file(weights_file)
    elif pytorch_path.exists():
        weights_file = pytorch_path
        state_dict = torch.load(weights_file, map_location="cpu")
    else:
        print("⚠️  No weight files found, using base model weights")
        return model
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    print(f"✓ Loaded weights from: {weights_file.name}")
    
    return model


def _load_peft_adapter_model(self):
    """Load standard PEFT adapter model."""
    print("Loading PEFT adapter model...")
    
    try:
        from peft import PeftModel, AutoPeftModelForCausalLM
    except ImportError:
        raise ImportError(
            "PEFT library required. Install with: pip install peft"
        )
    
    # Load using PEFT's auto loader
    model = AutoPeftModelForCausalLM.from_pretrained(
        self.model_path,
        torch_dtype=getattr(torch, self.dtype),
        device_map=self.device_map,
        trust_remote_code=self.trust_remote_code,
    )
    
    print(f"✓ Loaded PEFT adapter model")
    return model


def _load_merged_model(self):
    """Load merged PEFT model (adapters already merged into base)."""
    print("Loading merged model...")
    
    # Load as standard HuggingFace model
    model = AutoModelForCausalLM.from_pretrained(
        self.model_path,
        torch_dtype=getattr(torch, self.dtype),
        device_map=self.device_map,
        trust_remote_code=self.trust_remote_code,
        attn_implementation=self.attn_implementation,
    )
    
    print(f"✓ Loaded merged model")
    return model
```

### Step 3: Implement Tokenizer Setup

Add to `PeftMoEDecoder` class:

```python
def _setup_tokenizer(self):
    """Setup tokenizer for the model."""
    
    # Try loading tokenizer from model path first
    tokenizer_path = Path(self.model_path)
    
    if (tokenizer_path / "tokenizer.json").exists() or (tokenizer_path / "tokenizer_config.json").exists():
        print(f"Loading tokenizer from model path: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=self.trust_remote_code
        )
    else:
        # Load from base model
        print(f"Loading tokenizer from base model: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=self.trust_remote_code
        )
    
    # Set pad token if not set
    if self.tokenizer.pad_token is None:
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    self.skip_special_tokens = True
    
    print(f"✓ Tokenizer setup complete")
```

### Step 4: Implement Code Generation

Add to `PeftMoEDecoder` class:

```python
@torch.inference_mode()
def codegen(
    self, prompt: str, do_sample: bool = True, num_samples: int = 200
) -> List[str]:
    """
    Generate code completions using PEFT MoE model.
    
    Args:
        prompt: Input prompt
        do_sample: Use sampling (vs greedy)
        num_samples: Number of completions to generate
    
    Returns:
        List of generated code strings
    """
    if self.temperature == 0:
        assert not do_sample
        assert num_samples == 1
    
    # Prepare prompt (with chat template if applicable)
    prompt = (
        prompt
        if self.is_direct_completion()
        else make_raw_chat_prompt(
            prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
        )
    )
    
    # Tokenize
    input_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
    if self.device_map is None:
        input_tokens = input_tokens.to(self.device)
    
    # Setup generation kwargs
    gen_kwargs = {
        "max_new_tokens": self.max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": min(self.batch_size, num_samples),
        "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
    }
    
    if do_sample:
        gen_kwargs["top_p"] = 0.95
        gen_kwargs["temperature"] = self.temperature
    
    # Apply routing strategy
    self._apply_routing_strategy()
    
    # Generate
    outputs = self.model.generate(input_tokens, **gen_kwargs)
    
    # Decode
    gen_strs = self.tokenizer.batch_decode(
        outputs[:, input_tokens.size(-1):],
        skip_special_tokens=self.skip_special_tokens,
    )
    
    # Post-process: remove EOS tokens
    processed_outputs = []
    for output in gen_strs:
        min_index = len(output)
        for eos in self.eos:
            if eos in output:
                min_index = min(min_index, output.index(eos))
        processed_outputs.append(output[:min_index].replace("\t", "    "))
    
    return processed_outputs


def _apply_routing_strategy(self):
    """Apply routing strategy to model before generation."""
    
    # Check if model has expert management capabilities
    has_expert_manager = (
        hasattr(self.model, 'expert_manager') and 
        hasattr(self.model, 'router')
    )
    
    if not has_expert_manager:
        # No expert management - skip routing
        if self.routing_strategy != "router":
            print(f"⚠️  Model doesn't support routing, ignoring strategy: {self.routing_strategy}")
        return
    
    # Parse routing strategy
    if self.routing_strategy == "router":
        # Use model's internal router - activate all experts for MoE
        if self.model.expert_manager.num_experts > 1:
            self.model.expert_manager.activate_all_experts()
            print(f"Using router with {self.model.expert_manager.num_experts} experts")
    
    elif self.routing_strategy.startswith("single:"):
        # Single expert mode
        expert_id = int(self.routing_strategy.split(":")[1])
        if expert_id >= self.model.expert_manager.num_experts:
            raise ValueError(
                f"Expert {expert_id} not found. "
                f"Model has {self.model.expert_manager.num_experts} experts."
            )
        self.model.expert_manager.set_active_expert(expert_id)
        print(f"Using single expert: {expert_id}")
    
    elif self.routing_strategy == "best":
        # Analyze prompt and select best expert
        # For now, default to expert 0
        # TODO: Implement prompt analysis in future enhancement
        self.model.expert_manager.set_active_expert(0)
        print(f"Using 'best' strategy (selecting expert 0 for now)")
    
    elif self.routing_strategy == "ensemble":
        # Ensemble mode - activate all experts
        self.model.expert_manager.activate_all_experts()
        print(f"Using ensemble with {self.model.expert_manager.num_experts} experts")
    
    elif self.routing_strategy == "round_robin":
        # Round robin - for now, just use router
        # TODO: Implement round robin in future enhancement
        if self.model.expert_manager.num_experts > 1:
            self.model.expert_manager.activate_all_experts()
        print(f"Using round robin (defaulting to router for now)")
    
    else:
        raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")
```

### Step 5: Update Provider Factory

**File**: `evalplus/evalplus/provider/__init__.py`

Add to `make_model()` function:

```python
def make_model(
    model: str,
    backend: str,
    dataset: str,
    # ... existing params ...
    
    # New PEFT MoE params
    base_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
    routing_strategy: str = "router",
    wandb_artifact: Optional[str] = None,
    
    **kwargs,
) -> DecoderBase:
    """Create decoder with PEFT MoE support."""
    
    # Add peft_moe backend
    if backend == "peft_moe":
        from evalplus.provider.peft_moe import PeftMoEDecoder
        
        return PeftMoEDecoder(
            name=model,
            base_model=base_model,
            adapter_path=adapter_path,
            routing_strategy=routing_strategy,
            wandb_artifact=wandb_artifact,
            dataset=dataset,
            batch_size=batch_size,
            temperature=temperature,
            force_base_prompt=force_base_prompt,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            **kwargs,
        )
    
    # ... existing backends ...
```

## Testing

### Unit Tests

Create `tests/test_peft_moe_decoder.py`:

```python
import pytest
import torch
from pathlib import Path
from evalplus.provider.peft_moe import PeftMoEDecoder


def test_detect_dylora_moe_format(tmp_path):
    """Test DyLoRA-MoE format detection."""
    # Create mock DyLoRA-MoE directory
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    # Add config with dylora-moe marker
    config = {"model_type": "dylora-moe", "num_experts": 4}
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    decoder = PeftMoEDecoder(
        name=str(model_dir),
        dataset="humaneval"
    )
    
    assert decoder.model_format == "dylora_moe"


def test_detect_peft_adapter_format(tmp_path):
    """Test PEFT adapter format detection."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    # Add adapter_config.json
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "google/codegemma-2b"
    }
    with open(model_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f)
    
    decoder = PeftMoEDecoder(
        name=str(model_dir),
        dataset="humaneval"
    )
    
    assert decoder.model_format == "peft_adapter"


def test_routing_strategy_validation():
    """Test routing strategy validation."""
    with pytest.raises(ValueError, match="Unknown routing strategy"):
        decoder = PeftMoEDecoder(
            name="google/codegemma-2b",
            dataset="humaneval",
            routing_strategy="invalid_strategy"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_codegen_single_expert():
    """Test code generation with single expert."""
    decoder = PeftMoEDecoder(
        name="./path/to/model",
        base_model="google/codegemma-2b",
        dataset="humaneval",
        routing_strategy="single:0"
    )
    
    prompt = "def hello_world():"
    outputs = decoder.codegen(prompt, do_sample=False, num_samples=1)
    
    assert len(outputs) == 1
    assert isinstance(outputs[0], str)
```

### Integration Test

Create `tests/test_peft_moe_integration.py`:

```python
import pytest
from evalplus.codegen import run_codegen
from evalplus.evaluate import evaluate


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists("./trained_model"), reason="No trained model")
def test_full_evaluation_pipeline():
    """Test complete evaluation with PeftMoEDecoder."""
    
    # Generate code
    samples_path = run_codegen(
        model="./trained_model",
        backend="peft_moe",
        base_model="google/codegemma-2b",
        dataset="humaneval",
        routing_strategy="router",
        n_samples=1,
        greedy=True,
        id_range=[0, 10],  # Test on first 10 samples
    )
    
    # Evaluate
    results = evaluate(
        dataset="humaneval",
        samples=samples_path,
    )
    
    assert "pass@1" in results
    assert isinstance(results["pass@1"], float)
```

## Usage Examples

### CLI Usage

```bash
# Generate and evaluate with PEFT MoE model
evalplus.codegen \
    --model ./trained_model \
    --backend peft_moe \
    --base-model google/codegemma-2b \
    --routing-strategy router \
    --dataset humaneval \
    --greedy

# Evaluate specific expert
evalplus.codegen \
    --model ./trained_model \
    --backend peft_moe \
    --base-model google/codegemma-2b \
    --routing-strategy single:0 \
    --dataset humaneval

# W&B artifact
evalplus.codegen \
    --model user/project/model:v0 \
    --backend peft_moe \
    --wandb-artifact user/project/model:v0 \
    --base-model google/codegemma-2b \
    --dataset humaneval
```

### Python API Usage

```python
from evalplus.codegen import run_codegen
from evalplus.evaluate import evaluate

# Generate code with router
samples = run_codegen(
    model="./trained_model",
    backend="peft_moe",
    base_model="google/codegemma-2b",
    dataset="humaneval",
    routing_strategy="router",
    greedy=True,
    n_samples=1,
)

# Evaluate
results = evaluate(dataset="humaneval", samples=samples)
print(f"Pass@1: {results['pass@1']:.2f}%")

# Compare routing strategies
for strategy in ["router", "single:0", "single:1", "single:2"]:
    samples = run_codegen(
        model="./trained_model",
        backend="peft_moe",
        routing_strategy=strategy,
        dataset="humaneval",
        greedy=True,
    )
    results = evaluate(dataset="humaneval", samples=samples)
    print(f"{strategy}: {results['pass@1']:.2f}%")
```

## Validation Checklist

- [ ] Model format detection works for all formats
- [ ] W&B artifact loading works
- [ ] Base model inference from config works
- [ ] Tokenizer setup works
- [ ] All routing strategies work
- [ ] Code generation produces valid outputs
- [ ] EOS token handling works correctly
- [ ] Device placement works (CPU/CUDA)
- [ ] Error messages are helpful
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation is complete

## Next Steps

After completing Phase 1:
1. Test with real DyLoRA-MoE models
2. Proceed to Phase 2 (extend factory)
3. Proceed to Phase 3 (refactor benchmark.py)
4. Add expert analysis tools (Phase 4)
5. Complete documentation (Phase 5)

## Troubleshooting

### Common Issues

**ImportError: dylo_moe not found**
- Ensure DyLoRA-MoE package is in Python path
- Add to PYTHONPATH if needed

**Model format detection fails**
- Check config.json format
- Verify file structure matches expected format

**W&B artifact download fails**
- Check WANDB_API_KEY environment variable
- Verify artifact path format

**Generation produces empty outputs**
- Check EOS token configuration
- Verify model weights loaded correctly
- Check device placement

## Dependencies

Add to `evalplus/requirements.txt`:

```
# Existing dependencies
# ...

# PEFT MoE support (optional)
peft>=0.7.0  # For PEFT adapter loading
wandb>=0.16.0  # For W&B artifact support (optional)
safetensors>=0.4.0  # For efficient weight loading
```

## Documentation Updates

Update `evalplus/docs/cli.md`:

```markdown
## PEFT MoE Support

EvalPlus supports PEFT models with MoE architectures via the `peft_moe` backend.

### Usage

python -m evalplus.codegen \
    --model <model_path> \
    --backend peft_moe \
    --base-model <base_model_name> \
    --routing-strategy <strategy> \
    --dataset <dataset>

### Routing Strategies

- `router`: Use model's internal router (default)
- `single:<id>`: Use specific expert (e.g., `single:0`)
- `best`: Select best expert based on prompt
- `ensemble`: Combine all experts
- `round_robin`: Cycle through experts

### Examples

See [PEFT MoE Examples](./examples/peft_moe.md) for detailed examples.
```

This completes the Phase 1 implementation guide. The PeftMoEDecoder is now ready for implementation and testing.
