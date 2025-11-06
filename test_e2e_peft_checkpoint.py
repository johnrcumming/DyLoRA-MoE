"""
End-to-end test of PeftMoEDecoder with proper PEFT checkpoint from W&B.
Tests loading model with separated expert adapters and verifies MoE routing works.
"""

import sys
import os

workspace_root = r"w:\UnderTheRadar\DyLoRA-MoE"
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "evalplus"))

print("="*80)
print("E2E TEST: PeftMoEDecoder with W&B PEFT Checkpoint")
print("="*80)
print(f"Checkpoint: johnrcumming001/dylo-moe-full-training/best-dylora-model-full:v17")
print("="*80)

# Test 1: Load model from W&B artifact
print("\n[TEST 1] Loading model from W&B artifact...")
print("-"*80)

from evalplus.provider.peft_moe import PeftMoEDecoder

decoder = PeftMoEDecoder(
    name="dummy",  # Will be overridden by wandb_artifact
    dataset="humaneval",
    wandb_artifact="johnrcumming001/dylo-moe-full-training/best-dylora-model-full:v17",
    batch_size=1,
    temperature=0.0,
    dtype="bfloat16",
    routing_strategy="router",
    trust_remote_code=True,
    max_new_tokens=100
)

print("\n✓ Model loaded successfully!")
print(f"  Model format: {decoder.model_format}")
print(f"  Base model: {decoder.base_model_name}")
print(f"  Device: {decoder.device}")
print(f"  Routing strategy: {decoder.routing_strategy}")

# Test 2: Check if model has proper MoE structure
print("\n[TEST 2] Verifying MoE structure...")
print("-"*80)

if decoder.model_format == "dylora_moe":
    print("✓ Model format is dylora_moe (MoE capable)")
    
    # Check for expert manager
    if hasattr(decoder.model, 'expert_manager'):
        num_experts = decoder.model.expert_manager.num_experts
        print(f"✓ ExpertManager found: {num_experts} experts")
    else:
        print("✗ No ExpertManager found")
    
    # Check for router
    if hasattr(decoder.model, 'router'):
        print(f"✓ Router found: {type(decoder.model.router).__name__}")
    else:
        print("✗ No Router found")
else:
    print(f"⚠️  Model format is '{decoder.model_format}' (not MoE)")

# Test 3: Generate code with router strategy
print("\n[TEST 3] Testing code generation with router strategy...")
print("-"*80)

test_prompt = '''def fibonacci(n):
    """Return the nth Fibonacci number."""
'''

print(f"Prompt:\n{test_prompt}")
print("\nGenerating with router strategy...")

result = decoder.codegen(test_prompt, do_sample=False, num_samples=1)

print(f"\nGenerated code ({len(result[0])} chars):")
print("-"*80)
print(result[0][:500] if len(result[0]) > 500 else result[0])
if len(result[0]) > 500:
    print("... (truncated)")
print("-"*80)

# Check if output is meaningful (not just whitespace or repeating tokens)
meaningful = False
lines = result[0].strip().split('\n')
if len(lines) > 2:  # At least a few lines
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) > 1:
        meaningful = True

if meaningful:
    print("✓ Generation produced meaningful output")
else:
    print("✗ Generation produced empty or repetitive output")

# Test 4: Test different routing strategies
print("\n[TEST 4] Testing different routing strategies...")
print("-"*80)

strategies_to_test = ["router", "single:0", "single:1"]

for strategy in strategies_to_test:
    print(f"\nTesting strategy: {strategy}")
    
    # Create new decoder with different strategy
    decoder_test = PeftMoEDecoder(
        name="dummy",
        dataset="humaneval",
        wandb_artifact="johnrcumming001/dylo-moe-full-training/best-dylora-model-full:v17",
        batch_size=1,
        temperature=0.0,
        dtype="bfloat16",
        routing_strategy=strategy,
        trust_remote_code=True,
        max_new_tokens=50
    )
    
    short_prompt = "def add(a, b):\n    return"
    result = decoder_test.codegen(short_prompt, do_sample=False, num_samples=1)
    
    # Show first 100 chars
    output_preview = result[0][:100].replace('\n', '\\n')
    print(f"  Output: {output_preview}")
    
    # Check for repetition (common failure mode)
    tokens = result[0].split()
    if len(tokens) > 5:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio > 0.5:
            print(f"  ✓ Output diversity: {unique_ratio:.2%}")
        else:
            print(f"  ⚠️  Low diversity: {unique_ratio:.2%} (may be repetitive)")

# Test 5: HumanEval-style prompt
print("\n[TEST 5] Testing with HumanEval-style prompt...")
print("-"*80)

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

print("Prompt:")
print(humaneval_prompt)
print("\nGenerating...")

result = decoder.codegen(humaneval_prompt, do_sample=False, num_samples=1)

print(f"\nGenerated code ({len(result[0])} chars):")
print("-"*80)
print(result[0][:400] if len(result[0]) > 400 else result[0])
if len(result[0]) > 400:
    print("... (truncated)")
print("-"*80)

# Check if it looks like valid Python
has_indentation = '    ' in result[0]
has_keywords = any(kw in result[0] for kw in ['for', 'if', 'return', 'def'])

if has_indentation and has_keywords:
    print("✓ Output looks like valid Python code")
else:
    print("⚠️  Output may not be valid Python")

# Final summary
print("\n" + "="*80)
print("E2E TEST SUMMARY")
print("="*80)
print(f"Checkpoint: johnrcumming001/dylo-moe-full-training/best-dylora-model-full:v17")
print(f"Model format: {decoder.model_format}")
print(f"MoE capable: {'Yes' if decoder.model_format == 'dylora_moe' else 'No'}")
print(f"Generation working: {'Yes' if meaningful else 'Check output'}")
print("="*80)
