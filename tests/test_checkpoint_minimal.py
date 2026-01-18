#!/usr/bin/env python3
"""
Minimal DyLoRA-MoE Checkpoint Test
Tests checkpoint save/load without loading a second model (memory efficient).
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import json

class MinimalCheckpointTest:
    def __init__(self, model_name="google/codegemma-2b", num_experts=2):
        self.model_name = model_name
        self.num_experts = num_experts
        self.test_dir = None
        self.checkpoint_dir = None
        
    def setup(self):
        """Create test directory"""
        self.test_dir = tempfile.mkdtemp(prefix="dylora_checkpoint_minimal_")
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoint")
        print(f"\n{'='*80}")
        print(f"Test directory: {self.test_dir}")
        print(f"{'='*80}\n")
        
    def cleanup(self):
        """Remove test directory"""
        if self.test_dir and os.path.exists(self.test_dir):
            print(f"\nCleaning up: {self.test_dir}")
            shutil.rmtree(self.test_dir)
            
    def test_save_checkpoint(self):
        """Test saving a checkpoint"""
        print(f"{'='*80}")
        print("TEST: Save Checkpoint")
        print(f"{'='*80}")
        
        from dylo_moe.model import DyLoRA_MoE
        
        # Create model
        print("Creating model...")
        model = DyLoRA_MoE(
            model_name=self.model_name,
            num_experts=self.num_experts,
            lora_r=16,
            lora_alpha=32,
            balance_coefficient=0.01
        )
        print("✓ Model created")
        
        # Save checkpoint
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save PEFT adapters
        peft_model = model.expert_manager.model
        peft_dir = os.path.join(self.checkpoint_dir, "peft_adapters")
        
        for i in range(self.num_experts):
            adapter_name = f"expert_{i}"
            expert_dir = os.path.join(peft_dir, adapter_name)
            peft_model.save_pretrained(expert_dir, selected_adapters=[adapter_name])
            print(f"✓ Saved {adapter_name}")
            
        # Save router
        from safetensors.torch import save_file
        router_state = model.router.state_dict()
        router_path = os.path.join(self.checkpoint_dir, "router.safetensors")
        save_file(router_state, router_path)
        print(f"✓ Saved router")
        
        # Save config
        config = {
            "model_name": self.model_name,
            "num_experts": self.num_experts,
            "lora_r": 16,
            "lora_alpha": 32,
            "balance_coefficient": 0.01
        }
        config_path = os.path.join(self.checkpoint_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config.json")
        
        del model
        torch.cuda.empty_cache()
        
        return True
        
    def test_checkpoint_structure(self):
        """Verify checkpoint structure without loading model"""
        print(f"\n{'='*80}")
        print("TEST: Verify Checkpoint Structure")
        print(f"{'='*80}")
        
        checkpoint_path = Path(self.checkpoint_dir)
        peft_dir = checkpoint_path / "peft_adapters"
        
        if not peft_dir.exists():
            print(f"✗ Missing peft_adapters directory")
            return False
        print(f"✓ Found peft_adapters")
        
        # Check experts
        for i in range(self.num_experts):
            expert_dir = peft_dir / f"expert_{i}"
            if not expert_dir.exists():
                print(f"✗ Missing expert_{i} directory")
                return False
                
            # Handle nested PEFT directories
            nested_dir = expert_dir / f"expert_{i}"
            check_dir = nested_dir if nested_dir.exists() else expert_dir
            
            config_file = check_dir / "adapter_config.json"
            if not config_file.exists():
                print(f"✗ Missing adapter_config.json for expert_{i}")
                return False
                
            weights_file = check_dir / "adapter_model.safetensors"
            if not weights_file.exists():
                weights_file = check_dir / "adapter_model.bin"
                if not weights_file.exists():
                    print(f"✗ Missing adapter weights for expert_{i}")
                    return False
                    
            print(f"✓ Expert {i}: config + weights found")
            
        # Check router
        router_file = checkpoint_path / "router.safetensors"
        if not router_file.exists():
            router_file = checkpoint_path / "router.pt"
            if not router_file.exists():
                print(f"✗ Missing router state")
                return False
        print(f"✓ Router state found")
        
        # Check config
        config_file = checkpoint_path / "config.json"
        if not config_file.exists():
            print(f"✗ Missing config.json")
            return False
        print(f"✓ Config.json found")
        
        # Check for merged model files (should NOT exist)
        merged_files = [
            checkpoint_path / "model.safetensors",
            checkpoint_path / "pytorch_model.bin"
        ]
        for f in merged_files:
            if f.exists():
                print(f"✗ Found merged model file (should not exist): {f.name}")
                return False
        print(f"✓ No merged model files (correct)")
        
        return True
        
    def test_checkpoint_size(self):
        """Check checkpoint size is reasonable"""
        print(f"\n{'='*80}")
        print("TEST: Checkpoint Size")
        print(f"{'='*80}")
        
        total_size = 0
        checkpoint_path = Path(self.checkpoint_dir)
        
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                if size > 1024 * 1024:  # > 1MB
                    size_mb = size / (1024 * 1024)
                    print(f"  {file_path.relative_to(checkpoint_path)}: {size_mb:.1f} MB")
                    
        total_mb = total_size / (1024 * 1024)
        print(f"\nTotal checkpoint size: {total_mb:.1f} MB")
        
        if total_mb > 500:
            print(f"✗ Checkpoint too large (> 500 MB)")
            return False
            
        print(f"✓ Checkpoint size reasonable (< 500 MB)")
        return True
        
    def run_all_tests(self):
        """Run all tests"""
        print(f"\n{'='*80}")
        print("DyLoRA-MoE Minimal Checkpoint Tests")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"Experts: {self.num_experts}")
        print(f"{'='*80}")
        
        self.setup()
        
        results = {}
        
        try:
            # Test 1: Save checkpoint
            results['save'] = self.test_save_checkpoint()
            
            # Test 2: Verify structure
            results['structure'] = self.test_checkpoint_structure()
            
            # Test 3: Check size
            results['size'] = self.test_checkpoint_size()
            
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = False
            
        # Print summary
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
        print(f"{'='*80}")
        
        all_passed = all(results.values())
        if all_passed:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print(f"{'='*80}")
        
        print(f"\nℹ️  Checkpoint saved at: {self.checkpoint_dir}")
        print(f"   (Run with --keep-checkpoint to preserve)")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal DyLoRA-MoE checkpoint test")
    parser.add_argument("--model", default="google/codegemma-2b", help="Base model name")
    parser.add_argument("--experts", type=int, default=2, help="Number of experts")
    parser.add_argument("--keep-checkpoint", action="store_true", help="Don't delete checkpoint after test")
    
    args = parser.parse_args()
    
    tester = MinimalCheckpointTest(model_name=args.model, num_experts=args.experts)
    
    try:
        results = tester.run_all_tests()
        all_passed = all(results.values())
        
        if not args.keep_checkpoint:
            tester.cleanup()
        else:
            print(f"\n✓ Checkpoint preserved at: {tester.checkpoint_dir}")
            
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        tester.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        tester.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
