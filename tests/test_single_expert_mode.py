"""
Test single-expert training mode (expert_id parameter).
"""

import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dylo_moe.model import DyLoRA_MoE

print("=" * 80)
print("SINGLE EXPERT MODE TEST")
print("=" * 80)

# Create model
print("\nLoading model with 2 experts...")
model = DyLoRA_MoE(
    model_name="gpt2",  # 124M params - much faster for testing
    num_experts=2,
    lora_r=8,
    lora_alpha=16,
)
model.train()

# Test training with expert_id=0
print("\n--- Training Expert 0 Only ---")
input_ids = torch.randint(0, 1000, (2, 16))
labels = input_ids.clone()

outputs = model(input_ids=input_ids, labels=labels, expert_id=0)
print(f"Loss: {outputs.loss.item():.4f}")

outputs.loss.backward()

expert_0_grads = 0
expert_1_grads = 0

for name, param in model.foundation_model.named_parameters():
    if "lora" in name.lower() and param.grad is not None:
        if "expert_0" in name:
            expert_0_grads += 1
        elif "expert_1" in name:
            expert_1_grads += 1

print(f"\nExpert 0 params with gradients: {expert_0_grads}")
print(f"Expert 1 params with gradients: {expert_1_grads}")

assert expert_0_grads > 0, "❌ Expert 0 should have gradients!"
assert expert_1_grads == 0, "❌ Expert 1 should NOT have gradients!"

print("\n✅ PASS: Expert 0 training isolated correctly!")

# Test training with expert_id=1
print("\n--- Training Expert 1 Only ---")
model.zero_grad()

input_ids = torch.randint(0, 1000, (2, 16))
labels = input_ids.clone()

outputs = model(input_ids=input_ids, labels=labels, expert_id=1)
print(f"Loss: {outputs.loss.item():.4f}")

outputs.loss.backward()

expert_0_grads = 0
expert_1_grads = 0

for name, param in model.foundation_model.named_parameters():
    if "lora" in name.lower() and param.grad is not None:
        if "expert_0" in name:
            expert_0_grads += 1
        elif "expert_1" in name:
            expert_1_grads += 1

print(f"\nExpert 0 params with gradients: {expert_0_grads}")
print(f"Expert 1 params with gradients: {expert_1_grads}")

assert expert_0_grads == 0, "❌ Expert 0 should NOT have gradients!"
assert expert_1_grads > 0, "❌ Expert 1 should have gradients!"

print("\n✅ PASS: Expert 1 training isolated correctly!")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nSingle-expert training mode works correctly:")
print("  ✓ expert_id=0 trains only Expert 0")
print("  ✓ expert_id=1 trains only Expert 1")
print("  ✓ Gradient isolation prevents cross-expert leakage")
print("=" * 80)
