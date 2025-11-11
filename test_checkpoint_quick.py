#!/usr/bin/env python3
"""
Quick checkpoint test - minimal version for fast testing.

Tests basic save/load cycle without creating full models.
Useful for quick validation during development.
"""

import os
import sys
import json
import torch
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_peft_adapter_structure():
    """Test PEFT adapter structure can be created and validated."""
    print("\n" + "="*60)
    print("Quick Test: PEFT Adapter Structure")
    print("="*60)
    
    test_dir = tempfile.mkdtemp(prefix="quick_test_")
    checkpoint_dir = Path(test_dir) / "checkpoint"
    checkpoint_dir.mkdir()
    
    try:
        # Create minimal PEFT adapter structure
        for expert_id in range(2):
            expert_dir = checkpoint_dir / "peft_adapters" / f"expert_{expert_id}"
            expert_dir.mkdir(parents=True)
            
            # Create adapter_config.json
            config = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"]
            }
            with open(expert_dir / "adapter_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create dummy adapter weights
            dummy_weights = {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, 768),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(768, 16),
            }
            torch.save(dummy_weights, expert_dir / "adapter_model.bin")
            
            print(f"✓ Created expert_{expert_id} structure")
        
        # Create router state
        router_state = {
            "gate.weight": torch.randn(768, 2),
            "gate.bias": torch.randn(2)
        }
        torch.save(router_state, checkpoint_dir / "router.pt")
        print("✓ Created router state")
        
        # Create config.json
        config = {
            "model_type": "dylora-moe",
            "base_model_name_or_path": "google/codegemma-2b",
            "num_experts": 2,
            "checkpoint_type": "peft_only"
        }
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print("✓ Created config.json")
        
        # Validate structure
        print("\nValidating structure:")
        
        required_files = [
            "peft_adapters/expert_0/adapter_config.json",
            "peft_adapters/expert_0/adapter_model.bin",
            "peft_adapters/expert_1/adapter_config.json",
            "peft_adapters/expert_1/adapter_model.bin",
            "router.pt",
            "config.json"
        ]
        
        all_exist = True
        for file_path in required_files:
            full_path = checkpoint_dir / file_path
            exists = full_path.exists()
            symbol = "✓" if exists else "✗"
            print(f"{symbol} {file_path}")
            if not exists:
                all_exist = False
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in checkpoint_dir.rglob("*") if f.is_file())
        print(f"\nTotal size: {total_size / 1024:.1f} KB")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        if all_exist:
            print("\n✓ Quick test PASSED - structure is valid")
            return True
        else:
            print("\n✗ Quick test FAILED - missing files")
            return False
            
    except Exception as e:
        print(f"\n✗ Quick test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safetensors_availability():
    """Test if safetensors is available."""
    print("\n" + "="*60)
    print("Quick Test: Safetensors Availability")
    print("="*60)
    
    try:
        from safetensors.torch import save_file, load_file
        print("✓ safetensors is available")
        
        # Test save/load
        test_dir = tempfile.mkdtemp(prefix="safetensors_test_")
        test_file = Path(test_dir) / "test.safetensors"
        
        test_tensor = {"weight": torch.randn(10, 10)}
        save_file(test_tensor, test_file)
        loaded_tensor = load_file(test_file)
        
        if torch.allclose(test_tensor["weight"], loaded_tensor["weight"]):
            print("✓ safetensors save/load works")
            success = True
        else:
            print("✗ safetensors save/load failed")
            success = False
        
        import shutil
        shutil.rmtree(test_dir)
        return success
        
    except ImportError:
        print("✗ safetensors not available (will use PyTorch format)")
        print("  Install with: pip install safetensors")
        return False


def test_peft_availability():
    """Test if PEFT is available and has required methods."""
    print("\n" + "="*60)
    print("Quick Test: PEFT Availability")
    print("="*60)
    
    try:
        from peft import get_peft_model, LoraConfig, PeftModel
        print("✓ PEFT library is available")
        
        # Check for MoE support
        try:
            from peft import MoELoraConfig
            print("✓ MoELoraConfig is available")
        except ImportError:
            print("⚠️  MoELoraConfig not available (older PEFT version)")
        
        # Check for attach_router method
        if hasattr(PeftModel, 'attach_router'):
            print("✓ attach_router method is available")
        else:
            print("⚠️  attach_router method not available")
        
        return True
        
    except ImportError:
        print("✗ PEFT library not available")
        print("  Install with: pip install peft")
        return False


def main():
    """Run quick tests."""
    print("\n" + "="*60)
    print("DyLoRA-MoE Quick Checkpoint Tests")
    print("="*60)
    
    results = {}
    
    # Test 1: PEFT availability
    results["peft"] = test_peft_availability()
    
    # Test 2: Safetensors availability
    results["safetensors"] = test_safetensors_availability()
    
    # Test 3: Checkpoint structure
    results["structure"] = test_peft_adapter_structure()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    if all_passed:
        print("✓ ALL QUICK TESTS PASSED")
        print("\nReady to run full test: python test_checkpoint_save_load.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nFix issues before running full test")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
