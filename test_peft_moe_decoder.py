#!/usr/bin/env python3
"""
Test script for PeftMoEDecoder with real DyLoRA-MoE model.

This script tests that the PeftMoEDecoder can successfully:
1. Load a trained DyLoRA-MoE model
2. Generate code completions
3. Apply different routing strategies
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evalplus.provider.peft_moe import PeftMoEDecoder


def test_basic_loading():
    """Test basic model loading."""
    print("\n" + "="*80)
    print("TEST 1: Basic Model Loading")
    print("="*80)
    
    model_path = "./artifacts/best-dylora-model-full-v16/best_model"
    
    try:
        decoder = PeftMoEDecoder(
            name=model_path,
            dataset="humaneval",
            base_model="google/codegemma-2b",
            routing_strategy="router",
        )
        
        print("\n✅ Model loaded successfully!")
        print(f"   Format: {decoder.model_format}")
        print(f"   Base model: {decoder.base_model_name}")
        print(f"   Device: {decoder.device}")
        
        return decoder
    
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_code_generation(decoder):
    """Test code generation with a simple prompt."""
    print("\n" + "="*80)
    print("TEST 2: Code Generation")
    print("="*80)
    
    prompt = """def hello_world():
    \"\"\"Print hello world.\"\"\"
"""
    
    try:
        print(f"\nPrompt:\n{prompt}")
        print("\nGenerating (greedy)...")
        
        outputs = decoder.codegen(
            prompt=prompt,
            do_sample=False,  # Greedy
            num_samples=1
        )
        
        print(f"\n✅ Generation successful!")
        print(f"\nGenerated code:\n{outputs[0]}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_routing_strategies(decoder):
    """Test different routing strategies."""
    print("\n" + "="*80)
    print("TEST 3: Routing Strategies")
    print("="*80)
    
    # Check if model supports expert management
    has_expert_manager = hasattr(decoder.model, 'expert_manager')
    
    if not has_expert_manager:
        print("\n⚠️  Model doesn't have expert_manager - skipping routing tests")
        return
    
    num_experts = decoder.model.expert_manager.num_experts
    print(f"\nModel has {num_experts} experts")
    
    strategies = ["router"]
    
    # Add single expert strategies if we have multiple experts
    if num_experts > 1:
        strategies.extend([f"single:{i}" for i in range(min(num_experts, 3))])
    
    prompt = "def add(a, b):"
    
    for strategy in strategies:
        print(f"\n--- Testing strategy: {strategy} ---")
        
        try:
            # Update routing strategy
            decoder.routing_strategy = strategy
            
            # Generate
            outputs = decoder.codegen(
                prompt=prompt,
                do_sample=False,
                num_samples=1
            )
            
            print(f"✅ {strategy}: Generated {len(outputs[0])} chars")
            
        except Exception as e:
            print(f"❌ {strategy}: Failed - {e}")


def test_humaneval_format():
    """Test with actual HumanEval format prompt."""
    print("\n" + "="*80)
    print("TEST 4: HumanEval Format")
    print("="*80)
    
    model_path = "./artifacts/best-dylora-model-full-v16/best_model"
    
    # HumanEval prompt example
    prompt = """from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""
    
    try:
        decoder = PeftMoEDecoder(
            name=model_path,
            dataset="humaneval",
            base_model="google/codegemma-2b",
            routing_strategy="router",
            temperature=0.0,  # Greedy
        )
        
        print(f"\nPrompt:\n{prompt}")
        print("\nGenerating...")
        
        outputs = decoder.codegen(
            prompt=prompt,
            do_sample=False,
            num_samples=1
        )
        
        print(f"\n✅ Generation successful!")
        print(f"\nGenerated code:\n{outputs[0]}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PEFT MOE DECODER TEST SUITE")
    print("="*80)
    
    # Check if model exists
    model_path = "./artifacts/best-dylora-model-full-v16/best_model"
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at: {model_path}")
        print("   Please ensure the trained model is available.")
        return
    
    # Test 1: Basic loading
    decoder = test_basic_loading()
    if decoder is None:
        print("\n❌ Stopping tests - model loading failed")
        return
    
    # Test 2: Code generation
    success = test_code_generation(decoder)
    if not success:
        print("\n❌ Stopping tests - code generation failed")
        return
    
    # Test 3: Routing strategies
    test_routing_strategies(decoder)
    
    # Test 4: HumanEval format
    test_humaneval_format()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
