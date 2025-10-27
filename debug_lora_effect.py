"""
Test if LoRA outputs are actually affecting the loss.
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

input_ids = torch.randint(0, 1000, (2, 8))
attention_mask = torch.ones_like(input_ids)
labels = input_ids.clone()

print("\nForward pass 1 (with LoRA)...")
outputs1 = model(input_ids, attention_mask=attention_mask, labels=labels)
loss1 = outputs1.loss.item()
print(f"Loss with LoRA: {loss1:.4f}")

print("\nDisabling LoRA adapters...")
model.foundation_model.base_model.disable_adapters()

print("\nForward pass 2 (without LoRA)...")
# Need to bypass DyLoRA's router logic by using single expert mode
outputs2 = model.foundation_model(input_ids, attention_mask=attention_mask, labels=labels)
loss2 = outputs2.loss.item()
print(f"Loss without LoRA: {loss2:.4f}")

print(f"\nLoss difference: {abs(loss1 - loss2):.4f}")

if abs(loss1 - loss2) < 0.01:
    print("⚠️  LoRA doesn't seem to be affecting the loss!")
else:
    print("✓ LoRA is affecting the loss")
