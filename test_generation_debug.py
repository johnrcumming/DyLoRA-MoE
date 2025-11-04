"""Diagnostic test for PeftMoEDecoder generation."""

import sys
import os

workspace_root = r"w:\UnderTheRadar\DyLoRA-MoE"
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "evalplus"))

from evalplus.provider.peft_moe import PeftMoEDecoder
import torch

# Test instantiation
model_path = "./artifacts/best-dylora-model-full-v16/best_model"

print(f"Loading model from: {model_path}")
decoder = PeftMoEDecoder(
    name=model_path,
    dataset="humaneval",
    batch_size=1,
    temperature=0.0,
    dtype="bfloat16",
    routing_strategy="router",
    trust_remote_code=True,
    max_new_tokens=50  # Short generation for testing
)

print("\nModel loaded successfully!")
print(f"Tokenizer vocab size: {len(decoder.tokenizer)}")
print(f"Tokenizer pad token: {decoder.tokenizer.pad_token}")
print(f"Tokenizer eos token: {decoder.tokenizer.eos_token}")

# Test tokenization
test_prompt = "def add(a, b):\n    return"
tokens = decoder.tokenizer.encode(test_prompt)
print(f"\nTest prompt: {repr(test_prompt)}")
print(f"Encoded tokens: {tokens}")
print(f"Decoded: {repr(decoder.tokenizer.decode(tokens))}")

# Test direct model generation
print("\n" + "="*60)
print("Testing direct model generation...")
print("="*60)

input_ids = decoder.tokenizer.encode(test_prompt, return_tensors="pt").to(decoder.device)
print(f"Input shape: {input_ids.shape}")
print(f"Input tokens: {input_ids[0].tolist()}")

with torch.no_grad():
    outputs = decoder.model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=decoder.tokenizer.eos_token_id,
        eos_token_id=decoder.tokenizer.eos_token_id,
    )

print(f"Output shape: {outputs.shape}")
print(f"Output tokens: {outputs[0].tolist()}")

generated_text = decoder.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated text:\n{repr(generated_text)}")

# Now test through codegen method
print("\n" + "="*60)
print("Testing through codegen method...")
print("="*60)

result = decoder.codegen(test_prompt, do_sample=False, num_samples=1)
print(f"Result: {repr(result[0])}")
