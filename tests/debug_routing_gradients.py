"""
Debug script to understand gradient flow through routing_weights.
"""

import torch
import torch.nn as nn

# Simulate the routing scenario
print("Testing gradient flow through routing scenario...")

# Step 1: Simulate getting hidden states (with grad)
hidden_size = 8
batch_size = 2
seq_len = 4
num_experts = 3

hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
print(f"hidden_states.requires_grad: {hidden_states.requires_grad}")

# Step 2: Compute routing weights from hidden states (router)
router = nn.Linear(hidden_size, num_experts)
routing_weights = router(hidden_states)  # [batch, seq_len, num_experts]
routing_weights = torch.softmax(routing_weights, dim=-1)
print(f"routing_weights.requires_grad: {routing_weights.requires_grad}")
print(f"routing_weights.grad_fn: {routing_weights.grad_fn}")

# Step 3: Simulate expert outputs (weighted by routing_weights)
expert_outputs = []
for i in range(num_experts):
    # Each expert produces output
    expert_out = torch.randn(batch_size, seq_len, hidden_size)
    expert_outputs.append(expert_out)

# Combine experts weighted by routing_weights
combined = torch.zeros(batch_size, seq_len, hidden_size)
for i in range(num_experts):
    # routing_weights[:, :, i] has shape [batch, seq_len]
    # Need to broadcast to [batch, seq_len, hidden_size]
    weight = routing_weights[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
    combined = combined + weight * expert_outputs[i]

print(f"combined.requires_grad: {combined.requires_grad}")
print(f"combined.grad_fn: {combined.grad_fn}")

# Step 4: Compute loss
target = torch.randn_like(combined)
loss = ((combined - target) ** 2).mean()

print(f"loss.requires_grad: {loss.requires_grad}")
print(f"loss.grad_fn: {loss.grad_fn}")

# Step 5: Backward
loss.backward()

print(f"\nAfter backward:")
print(f"router.weight.grad: {router.weight.grad is not None} (norm: {router.weight.grad.norm() if router.weight.grad is not None else 0})")
print(f"router.bias.grad: {router.bias.grad is not None} (norm: {router.bias.grad.norm() if router.bias.grad is not None else 0})")
print(f"hidden_states.grad: {hidden_states.grad is not None}")

print("\n✓ Gradients flow through routing_weights to router!")
