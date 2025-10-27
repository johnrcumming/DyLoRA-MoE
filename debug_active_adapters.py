"""
Debug which adapters are active and how routing_weights are being used.
"""

import torch
import os
from dylo_moe.model import DyLoRA_MoE

model_name = "google/gemma-3-270m-it"

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        with open("hf_token.txt", "r") as f:
            hf_token = f.read().strip()
    except FileNotFoundError:
        exit(1)

print("Creating model...")
model = DyLoRA_MoE(
    model_name=model_name,
    num_experts=4,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    token=hf_token,
    allow_expert_growth=False,
    balance_coefficient=0.0,
)
model.train()

# Check active adapters
print("\nChecking active adapters before forward pass...")
if hasattr(model.foundation_model, 'base_model'):
    base = model.foundation_model.base_model
    if hasattr(base, 'active_adapters'):
        print(f"  active_adapters: {base.active_adapters}")
    
    # Find a LoRA layer to inspect
    for name, module in base.named_modules():
        if hasattr(module, 'active_adapters') and 'lora' in name.lower():
            print(f"  Layer {name}:")
            print(f"    active_adapters: {module.active_adapters}")
            print(f"    lora_A keys: {list(module.lora_A.keys()) if hasattr(module, 'lora_A') else 'N/A'}")
            break

print("\nCreating input...")
input_ids = torch.randint(0, 1000, (2, 8))
attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()

print("\nRunning forward pass...")
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

print(f"Loss: {outputs.loss.item():.4f}")

# Check active adapters after forward pass
print("\nChecking active adapters after forward pass...")
if hasattr(model.foundation_model, 'base_model'):
    base = model.foundation_model.base_model
    if hasattr(base, 'active_adapters'):
        print(f"  active_adapters: {base.active_adapters}")
    
    # Find a LoRA layer to inspect
    for name, module in base.named_modules():
        if hasattr(module, 'active_adapters') and 'lora' in name.lower():
            print(f"  Layer {name}:")
            print(f"    active_adapters: {module.active_adapters}")
            break

print("\nChecking routing_weights shape...")
if model.last_routing_weights is not None:
    print(f"  routing_weights shape: {model.last_routing_weights.shape}")
    print(f"  Expected: [batch=2, seq=8, num_experts=4]")
