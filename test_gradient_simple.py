#!/usr/bin/env python3
"""
Simpler gradient flow test for routing_weights
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


def test_simple_gradient_flow():
    """Test gradient flow with a simpler setup"""
    print("Testing gradient flow through routing_weights...")
    
    # Create a very simple model (just embeddings + linear)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 32)
            
        def forward(self, x, **kwargs):
            x = self.embed(x)
            x = self.linear(x, **kwargs)  # Pass kwargs to linear
            return x.mean()  # Simple scalar output
    
    model = SimpleModel()
    
    # Add LoRA to the linear layer
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["linear"],
        lora_dropout=0.0,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    model.add_adapter("expert_1", lora_config)
    model.base_model.set_adapter(["expert_0", "expert_1"])
    
    # Create input
    input_ids = torch.randint(0, 100, (2, 5))
    
    # Create routing weights with grad
    routing_logits = torch.randn(2, 5, 2, requires_grad=True)
    routing_weights = torch.nn.functional.softmax(routing_logits, dim=-1)
    
    # Forward
    output = model(input_ids, routing_weights=routing_weights)
    print(f"Output: {output.item():.4f}")
    print(f"Routing weights in graph: {routing_weights.grad_fn is not None}")
    
    # Backward
    output.backward()
    
    # Check gradient
    if routing_logits.grad is not None:
        print(f"✅ Gradient exists! Norm: {routing_logits.grad.norm().item():.6f}")
        return True
    else:
        print(f"❌ No gradient!")
        return False


if __name__ == "__main__":
    success = test_simple_gradient_flow()
    exit(0 if success else 1)
