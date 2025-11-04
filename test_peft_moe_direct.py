"""
Direct test of PeftMoEDecoder without module import chain.
Loads the code directly to bypass import issues.
"""

import sys
import os

# Set paths
workspace_root = r"w:\UnderTheRadar\DyLoRA-MoE"
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "evalplus"))

print("Testing direct imports...")
print("=" * 60)

# Test basic transformers imports
print("\n1. Testing transformers imports...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("   ✓ transformers imports successful")
except Exception as e:
    print(f"   ✗ transformers import failed: {e}")
    sys.exit(1)

# Test PEFT imports
print("\n2. Testing PEFT imports...")
try:
    from peft import PeftModel, get_peft_model, LoraConfig, AutoPeftModelForCausalLM
    print("   ✓ PEFT imports successful")
except Exception as e:
    print(f"   ✗ PEFT import failed: {e}")
    sys.exit(1)

# Test dylo_moe imports
print("\n3. Testing dylo_moe imports...")
try:
    from dylo_moe.model import DyLoRA_MoE
    from dylo_moe.expert import ExpertManager
    from dylo_moe.router import DynamicHybridRouter
    print("   ✓ dylo_moe imports successful")
except Exception as e:
    print(f"   ✗ dylo_moe import failed: {e}")
    sys.exit(1)

# Test torch
print("\n4. Testing torch import...")
try:
    import torch
    print(f"   ✓ torch {torch.__version__} imported")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"   ✗ torch import failed: {e}")
    sys.exit(1)

# Now test the PeftMoEDecoder by loading it directly
print("\n5. Testing PeftMoEDecoder import...")
try:
    # Import by adding evalplus to path
    from evalplus.provider.peft_moe import PeftMoEDecoder
    print("   ✓ PeftMoEDecoder imported successfully")
except Exception as e:
    print(f"   ✗ PeftMoEDecoder import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test instantiation
print("\n6. Testing PeftMoEDecoder instantiation...")
model_path = "./artifacts/best-dylora-model-full-v16/best_model"

if not os.path.exists(model_path):
    print(f"   ✗ Model not found at {model_path}")
    sys.exit(1)

print(f"   Model path exists: {model_path}")

try:
    decoder = PeftMoEDecoder(
        name=model_path,
        dataset="humaneval",  # Required parameter for EOS tokens
        batch_size=1,
        temperature=0.0,
        dtype="bfloat16",
        routing_strategy="router",
        trust_remote_code=True
    )
    print("   ✓ PeftMoEDecoder instantiated successfully")
    print(f"   ✓ Model format detected: {decoder.model_format}")
    print(f"   ✓ Model loaded on device: {decoder.device}")
except Exception as e:
    print(f"   ✗ PeftMoEDecoder instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test code generation
print("\n7. Testing code generation...")
test_prompt = '''def add(a, b):
    """Add two numbers and return the result."""
'''

try:
    result = decoder.codegen(test_prompt, do_sample=False, num_samples=1)
    print("   ✓ Code generation successful")
    print(f"   Generated {len(result)} sample(s)")
    if result and result[0]:
        print(f"   Output length: {len(result[0])} characters")
        print(f"   First 200 chars:\n   {result[0][:200]}")
    else:
        print(f"   WARNING: Empty output received")
except Exception as e:
    print(f"   ✗ Code generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
