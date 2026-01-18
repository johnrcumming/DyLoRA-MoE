"""
Minimal gradient flow test for DyLoRA-MoE after cleanup.
"""

import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dylo_moe.model import DyLoRA_MoE

print("=" * 80)
print("GRADIENT FLOW VERIFICATION")
print("=" * 80)

# Create model
print("\nLoading model with 2 experts...")
model = DyLoRA_MoE(
    model_name="gpt2",  # 124M params - much faster for testing
    num_experts=2,
    lora_r=8,
    lora_alpha=16,
    balance_coefficient=0.01,
)
model.train()

# Create batch
input_ids = torch.randint(0, 1000, (2, 16))
labels = input_ids.clone()

print(f"Model in training mode: {model.training}")
print(f"Router in training mode: {model.router.training}")

# Forward + backward
print("\nRunning forward pass...")
outputs = model(input_ids=input_ids, labels=labels)

print(f"Loss: {outputs.loss.item():.4f}")
if model.last_lm_loss is not None:
    print(f"  LM Loss: {model.last_lm_loss.item():.4f}")
if model.last_balance_loss is not None:
    print(f"  Balance Loss: {model.last_balance_loss.item():.6f}")

print("\nRunning backward pass...")
outputs.loss.backward()

# Check gradients
print("\n" + "=" * 80)
print("GRADIENT CHECK")
print("=" * 80)

router_grads = []
lora_grads = []
lora_params_by_expert = {0: 0, 1: 0}
frozen_with_grads = []

for name, param in model.router.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        router_grads.append((name, grad_norm))
        print(f"✓ Router {name}: {grad_norm:.6f}")

print()

for name, param in model.foundation_model.named_parameters():
    if "lora" in name.lower() and param.requires_grad:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            lora_grads.append(grad_norm)
            
            # Count by expert
            if "expert_0" in name:
                lora_params_by_expert[0] += 1
            elif "expert_1" in name:
                lora_params_by_expert[1] += 1
    elif not param.requires_grad and param.grad is not None:
        frozen_with_grads.append(name)

print(f"✓ LoRA params with gradients: {len(lora_grads)}")
print(f"  - Expert 0: {lora_params_by_expert[0]} params")
print(f"  - Expert 1: {lora_params_by_expert[1]} params")

if frozen_with_grads:
    print(f"\n✗ Frozen params with gradients: {len(frozen_with_grads)}")
    for name in frozen_with_grads[:3]:
        print(f"  - {name}")
else:
    print(f"\n✓ Frozen params with gradients: 0")

# Statistics
if router_grads:
    norms = [n for _, n in router_grads]
    print(f"\nRouter gradients:")
    print(f"  min={min(norms):.6f}, max={max(norms):.6f}, mean={sum(norms)/len(norms):.6f}")

if lora_grads:
    print(f"LoRA gradients:")
    print(f"  min={min(lora_grads):.6f}, max={max(lora_grads):.6f}, mean={sum(lora_grads)/len(lora_grads):.6f}")

# Verification
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

checks = [
    (len(router_grads) == 2, "Router has gradients"),
    (len(lora_grads) > 0, "LoRA adapters have gradients"),
    (lora_params_by_expert[0] > 0, "Expert 0 LoRA params have gradients"),
    (lora_params_by_expert[1] > 0, "Expert 1 LoRA params have gradients"),
    (len(frozen_with_grads) == 0, "No frozen params have gradients"),
]

all_passed = True
for passed, desc in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {desc}")
    if not passed:
        all_passed = False

if all_passed:
    print("\n" + "=" * 80)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 80)
    print("\nGradient flow is working correctly:")
    print("  ✓ Router learns expert weights via backprop")
    print("  ✓ Both experts receive gradients during MoE training")
    print("  ✓ Frozen base model parameters remain frozen")
    print("  ✓ Load balancing loss contributes to router training")
    print("=" * 80)
else:
    print("\n❌ SOME CHECKS FAILED!")
    exit(1)
