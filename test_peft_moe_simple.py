"""Simple test of PeftMoEDecoder code generation."""

import sys
import os

workspace_root = r"w:\UnderTheRadar\DyLoRA-MoE"
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "evalplus"))

from evalplus.provider.peft_moe import PeftMoEDecoder

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
    trust_remote_code=True
)

print("\nModel loaded successfully!")
print(f"Format: {decoder.model_format}")
print(f"Device: {decoder.device}")

# Test code generation
test_prompt = '''def add(a, b):
    """Add two numbers and return the result."""
'''

print("\n" + "="*60)
print("Testing code generation...")
print("="*60)
print(f"Prompt:\n{test_prompt}")

result = decoder.codegen(test_prompt, do_sample=False, num_samples=1)

print(f"\nGenerated {len(result)} sample(s)")
print(f"Output length: {len(result[0])} characters")
print("\n" + "="*60)
print("Generated code:")
print("="*60)
print(result[0])
print("="*60)

# Test with HumanEval-style prompt
humaneval_prompt = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

print("\n\n" + "="*60)
print("Testing HumanEval-style prompt...")
print("="*60)
print(f"Prompt:\n{humaneval_prompt}")

result2 = decoder.codegen(humaneval_prompt, do_sample=False, num_samples=1)

print(f"\nGenerated {len(result2)} sample(s)")
print(f"Output length: {len(result2[0])} characters")
print("\n" + "="*60)
print("Generated code:")
print("="*60)
print(result2[0])
print("="*60)

print("\n\nAll tests completed successfully!")
