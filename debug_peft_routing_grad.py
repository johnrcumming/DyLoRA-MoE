"""
Test if routing_weights gradients flow in actual PEFT forward pass.
"""

import torch
import os
from dylo_moe.model import DyLoRA_MoE

# Use tiny model
model_name = "google/gemma-3-270m-it"

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        with open("hf_token.txt", "r") as f:
            hf_token = f.read().strip()
    except FileNotFoundError:
        print("❌ Could not find HF token")
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
    balance_coefficient=0.0,  # Disable balance loss to isolate routing gradients
)
model.train()

print("Creating input...")
input_ids = torch.randint(0, 1000, (2, 16))
attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()

print("\nRunning forward pass...")

# Monkey-patch to capture routing_weights WITH gradients
routing_weights_captured = None

original_forward = model.foundation_model.forward

def capture_routing_weights(self, *args, **kwargs):
    global routing_weights_captured
    if 'routing_weights' in kwargs and kwargs['routing_weights'] is not None:
        routing_weights_captured = kwargs['routing_weights']
        print(f"  Captured routing_weights:")
        print(f"    - requires_grad: {routing_weights_captured.requires_grad}")
        print(f"    - grad_fn: {routing_weights_captured.grad_fn}")
    return original_forward(*args, **kwargs)

model.foundation_model.forward = lambda *args, **kwargs: capture_routing_weights(model.foundation_model, *args, **kwargs)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

print(f"\nLoss: {outputs.loss.item():.4f}")
print(f"Loss requires_grad: {outputs.loss.requires_grad}")
print(f"Loss grad_fn: {outputs.loss.grad_fn}")

if routing_weights_captured is not None:
    print(f"\nRouting weights (captured during forward):")
    print(f"  - requires_grad: {routing_weights_captured.requires_grad}")
    print(f"  - grad_fn: {routing_weights_captured.grad_fn}")
    print(f"  - is_leaf: {routing_weights_captured.is_leaf}")

print("\nRunning backward...")
model.zero_grad()
outputs.loss.backward()

print("\nChecking router gradients...")
for name, param in model.router.named_parameters():
    if param.grad is not None:
        print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")
    else:
        print(f"  {name}: NO GRADIENT")

if routing_weights_captured is not None and routing_weights_captured.grad is not None:
    print(f"\nrouting_weights.grad: {routing_weights_captured.grad.norm().item():.6f}")
else:
    print(f"\nrouting_weights.grad: None")
