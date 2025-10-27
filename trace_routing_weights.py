#!/usr/bin/env python3
"""
Debug script to trace routing_weights propagation
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


# Monkey patch to trace calls
original_linear_forward = None

def trace_linear_forward(self, x, *args, **kwargs):
    if 'routing_weights' in kwargs:
        print(f"  ✓ routing_weights received in {self.__class__.__name__}")
        print(f"    Shape: {kwargs['routing_weights'].shape}")
    else:
        print(f"  ✗ NO routing_weights in {self.__class__.__name__}")
    return original_linear_forward(self, x, *args, **kwargs)


def main():
    print("Tracing routing_weights propagation through model...\n")
    
    # Create model
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    model.add_adapter("expert_1", lora_config)
    model.base_model.set_adapter(["expert_0", "expert_1"])
    
    # Patch Linear.forward
    from peft.tuners.lora.layer import Linear
    global original_linear_forward
    original_linear_forward = Linear.forward
    Linear.forward = trace_linear_forward
    
    # Create input
    input_ids = torch.randint(0, 1000, (1, 5))
    routing_weights = torch.randn(1, 5, 2)
    
    print("Calling model with routing_weights...\n")
    try:
        outputs = model(input_ids, routing_weights=routing_weights)
        print(f"\n✅ Forward completed successfully")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
