#!/usr/bin/env python3
"""
Test script for DyLoRA-MoE checkpoint saving and loading.

Tests:
1. Save complete MoE-PEFT checkpoint (adapters + router + skill library)
2. Load checkpoint and verify all components restored correctly
3. Verify model outputs are identical before/after save/load
4. Test both PEFT adapter loading and router state loading
5. Verify checkpoint structure follows PEFT conventions
"""

import os
import sys
import json
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dylo_moe.model import DyLoRA_MoE
from transformers import AutoTokenizer


class CheckpointTester:
    """Test checkpoint save/load functionality."""
    
    def __init__(self, model_name: str = "google/codegemma-2b", num_experts: int = 2):
        self.model_name = model_name
        self.num_experts = num_experts
        self.test_dir = None
        self.results = []
        
    def log(self, message: str, success: bool = True):
        """Log test result."""
        symbol = "✓" if success else "✗"
        print(f"{symbol} {message}")
        self.results.append({"message": message, "success": success})
        
    def test_checkpoint_structure(self, checkpoint_dir: str) -> bool:
        """Verify checkpoint directory has correct structure."""
        print("\n" + "="*80)
        print("TEST 1: Verify Checkpoint Structure")
        print("="*80)
        
        checkpoint_path = Path(checkpoint_dir)
        
        # Check for PEFT adapters directory
        peft_dir = checkpoint_path / "peft_adapters"
        if not peft_dir.exists():
            self.log(f"Missing peft_adapters directory", False)
            return False
        self.log(f"Found peft_adapters directory: {peft_dir}")
        
        # Check each expert adapter
        for expert_id in range(self.num_experts):
            expert_dir = peft_dir / f"expert_{expert_id}"
            if not expert_dir.exists():
                self.log(f"Missing expert_{expert_id} directory", False)
                return False
            self.log(f"Found expert_{expert_id} directory")
            
            # PEFT creates nested directories: expert_0/expert_0/adapter_config.json
            # Check both locations for compatibility
            nested_dir = expert_dir / f"expert_{expert_id}"
            check_dir = nested_dir if nested_dir.exists() else expert_dir
            
            # Check for adapter_config.json
            config_file = check_dir / "adapter_config.json"
            if not config_file.exists():
                self.log(f"Missing adapter_config.json for expert_{expert_id} (checked {check_dir})", False)
                return False
            self.log(f"Found adapter_config.json for expert_{expert_id}")
            
            # Check for adapter weights (either safetensors or bin)
            safetensors_file = check_dir / "adapter_model.safetensors"
            bin_file = check_dir / "adapter_model.bin"
            if not (safetensors_file.exists() or bin_file.exists()):
                self.log(f"Missing adapter weights for expert_{expert_id}", False)
                return False
            weights_file = safetensors_file if safetensors_file.exists() else bin_file
            self.log(f"Found adapter weights: {weights_file.name}")
        
        # Check for router
        router_safetensors = checkpoint_path / "router.safetensors"
        router_pt = checkpoint_path / "router.pt"
        if not (router_safetensors.exists() or router_pt.exists()):
            self.log("Missing router state", False)
            return False
        router_file = router_safetensors if router_safetensors.exists() else router_pt
        self.log(f"Found router state: {router_file.name}")
        
        # Check for config.json
        config_file = checkpoint_path / "config.json"
        if not config_file.exists():
            self.log("Missing config.json", False)
            return False
        self.log(f"Found config.json")
        
        # Validate config.json content
        with open(config_file) as f:
            config = json.load(f)
            if config.get("model_type") != "dylora-moe":
                self.log("Invalid model_type in config.json", False)
                return False
            if config.get("num_experts") != self.num_experts:
                self.log(f"Config num_experts mismatch: {config.get('num_experts')} != {self.num_experts}", False)
                return False
        self.log("Config.json validated")
        
        return True
    
    def test_save_checkpoint(self, model: DyLoRA_MoE, checkpoint_dir: str) -> bool:
        """Test saving a checkpoint."""
        print("\n" + "="*80)
        print("TEST 2: Save Checkpoint")
        print("="*80)
        
        try:
            # Save PEFT adapters
            peft_model = model.expert_manager.model
            peft_adapters_dir = os.path.join(checkpoint_dir, "peft_adapters")
            
            for expert_id in range(self.num_experts):
                adapter_name = f"expert_{expert_id}"
                expert_dir = os.path.join(peft_adapters_dir, adapter_name)
                
                peft_model.set_adapter(adapter_name)
                peft_model.save_pretrained(expert_dir, selected_adapters=[adapter_name])
                self.log(f"Saved expert_{expert_id} adapter")
            
            # Save router
            router_path = os.path.join(checkpoint_dir, "router.safetensors")
            try:
                from safetensors.torch import save_file
                router_state = model.router.state_dict()
                save_file(router_state, router_path)
                self.log(f"Saved router (safetensors)")
            except ImportError:
                router_path = os.path.join(checkpoint_dir, "router.pt")
                torch.save(model.router.state_dict(), router_path)
                self.log(f"Saved router (PyTorch)")
            
            # Save skill library (if it has state)
            if hasattr(model, 'skill_library'):
                skill_path = os.path.join(checkpoint_dir, "skill_library.pt")
                try:
                    # SkillLibrary has save() method, not state_dict()
                    model.skill_library.save(skill_path)
                    self.log(f"Saved skill library")
                except Exception as e:
                    self.log(f"Skill library save skipped: {e}")
            
            # Save config
            config_path = os.path.join(checkpoint_dir, "config.json")
            base_model_name = getattr(model.foundation_model.config, '_name_or_path', self.model_name)
            
            config = {
                "model_type": "dylora-moe",
                "_dylora_format": "peft_separated",
                "base_model_name_or_path": base_model_name,
                "num_experts": self.num_experts,
                "checkpoint_type": "peft_only",
                "_note": "Test checkpoint for DyLoRA-MoE"
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.log("Saved config.json")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to save checkpoint: {e}", False)
            import traceback
            traceback.print_exc()
            return False
    
    def test_load_checkpoint(self, checkpoint_dir: str, hf_token: str = None) -> DyLoRA_MoE:
        """Test loading a checkpoint."""
        print("\n" + "="*80)
        print("TEST 3: Load Checkpoint")
        print("="*80)
        
        try:
            # Read config
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path) as f:
                config = json.load(f)
            
            base_model_name = config.get("base_model_name_or_path", self.model_name)
            num_experts = config.get("num_experts", self.num_experts)
            
            self.log(f"Loading model: {base_model_name}")
            self.log(f"Number of experts: {num_experts}")
            
            # Create new model instance
            model = DyLoRA_MoE(
                base_model_name,
                num_experts=num_experts,
                token=hf_token,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                allow_expert_growth=False,
                balance_coefficient=0.01
            )
            self.log("Created new model instance")
            
            # Load PEFT adapters
            peft_model = model.expert_manager.model
            peft_adapters_dir = os.path.join(checkpoint_dir, "peft_adapters")
            
            for expert_id in range(num_experts):
                adapter_name = f"expert_{expert_id}"
                expert_dir = os.path.join(peft_adapters_dir, adapter_name)
                
                # PEFT may create nested directories: expert_0/expert_0/
                nested_dir = os.path.join(expert_dir, adapter_name)
                load_dir = nested_dir if os.path.exists(nested_dir) else expert_dir
                
                if expert_id == 0:
                    # First adapter is already created, just load weights
                    from peft import PeftModel
                    peft_model.load_adapter(load_dir, adapter_name=adapter_name)
                else:
                    # Load additional adapters
                    peft_model.load_adapter(load_dir, adapter_name=adapter_name)
                
                self.log(f"Loaded expert_{expert_id} adapter")
            
            # Load router
            router_safetensors = os.path.join(checkpoint_dir, "router.safetensors")
            router_pt = os.path.join(checkpoint_dir, "router.pt")
            
            if os.path.exists(router_safetensors):
                try:
                    from safetensors.torch import load_file
                    router_state = load_file(router_safetensors)
                    model.router.load_state_dict(router_state)
                    self.log("Loaded router (safetensors)")
                except ImportError:
                    router_state = torch.load(router_pt, map_location='cpu')
                    model.router.load_state_dict(router_state)
                    self.log("Loaded router (PyTorch)")
            elif os.path.exists(router_pt):
                router_state = torch.load(router_pt, map_location='cpu')
                model.router.load_state_dict(router_state)
                self.log("Loaded router (PyTorch)")
            else:
                self.log("No router state found", False)
            
            return model
            
        except Exception as e:
            self.log(f"Failed to load checkpoint: {e}", False)
            import traceback
            traceback.print_exc()
            return None
    
    def test_output_consistency(self, model1: DyLoRA_MoE, model2: DyLoRA_MoE, 
                                tokenizer: AutoTokenizer) -> bool:
        """Test that outputs are identical before and after save/load."""
        print("\n" + "="*80)
        print("TEST 4: Output Consistency")
        print("="*80)
        
        try:
            # Create test input
            test_prompt = "def factorial(n):\n    "
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            # Get outputs from original model
            model1.eval()
            with torch.no_grad():
                outputs1 = model1(**inputs)
                logits1 = outputs1.logits
            
            # Get outputs from loaded model
            model2.eval()
            with torch.no_grad():
                outputs2 = model2(**inputs)
                logits2 = outputs2.logits
            
            # Compare logits
            if logits1.shape != logits2.shape:
                self.log(f"Shape mismatch: {logits1.shape} != {logits2.shape}", False)
                return False
            self.log(f"Output shapes match: {logits1.shape}")
            
            # Check numerical similarity
            max_diff = torch.abs(logits1 - logits2).max().item()
            mean_diff = torch.abs(logits1 - logits2).mean().item()
            
            self.log(f"Max logit difference: {max_diff:.6e}")
            self.log(f"Mean logit difference: {mean_diff:.6e}")
            
            # Tolerance for floating point comparison
            tolerance = 1e-5
            if max_diff > tolerance:
                self.log(f"Outputs differ by more than {tolerance}", False)
                return False
            
            self.log("Outputs are identical (within tolerance)")
            
            # Compare router states
            router_state1 = model1.router.state_dict()
            router_state2 = model2.router.state_dict()
            
            for key in router_state1.keys():
                if key not in router_state2:
                    self.log(f"Router state key missing: {key}", False)
                    return False
                
                diff = torch.abs(router_state1[key] - router_state2[key]).max().item()
                if diff > tolerance:
                    self.log(f"Router state differs for {key}: {diff}", False)
                    return False
            
            self.log("Router states match")
            
            return True
            
        except Exception as e:
            self.log(f"Failed consistency test: {e}", False)
            import traceback
            traceback.print_exc()
            return False
    
    def test_checkpoint_size(self, checkpoint_dir: str) -> bool:
        """Test that checkpoint is reasonably sized (no merged model)."""
        print("\n" + "="*80)
        print("TEST 5: Checkpoint Size")
        print("="*80)
        
        try:
            checkpoint_path = Path(checkpoint_dir)
            total_size = 0
            
            for file_path in checkpoint_path.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    total_size += size
                    size_mb = size / (1024 * 1024)
                    if size_mb > 1:  # Only print files > 1MB
                        print(f"  {file_path.relative_to(checkpoint_path)}: {size_mb:.1f} MB")
            
            total_size_mb = total_size / (1024 * 1024)
            self.log(f"Total checkpoint size: {total_size_mb:.1f} MB")
            
            # Check for merged model files (should not exist)
            merged_files = [
                checkpoint_path / "model.safetensors",
                checkpoint_path / "pytorch_model.bin"
            ]
            
            for merged_file in merged_files:
                if merged_file.exists():
                    size_mb = merged_file.stat().st_size / (1024 * 1024)
                    self.log(f"Found merged model file: {merged_file.name} ({size_mb:.1f} MB)", False)
                    return False
            
            self.log("No merged model files found (correct)")
            
            # Checkpoint should be < 500MB for 2 experts (generous threshold)
            max_size_mb = 500
            if total_size_mb > max_size_mb:
                self.log(f"Checkpoint too large: {total_size_mb:.1f} MB > {max_size_mb} MB", False)
                return False
            
            self.log(f"Checkpoint size is reasonable (< {max_size_mb} MB)")
            return True
            
        except Exception as e:
            self.log(f"Failed size test: {e}", False)
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self, hf_token: str = None) -> Dict[str, Any]:
        """Run all checkpoint tests."""
        print("\n" + "="*80)
        print("DyLoRA-MoE Checkpoint Save/Load Tests")
        print("="*80)
        print(f"Model: {self.model_name}")
        print(f"Experts: {self.num_experts}")
        print("="*80)
        
        # Create temporary directory for test
        self.test_dir = tempfile.mkdtemp(prefix="dylora_checkpoint_test_")
        checkpoint_dir = os.path.join(self.test_dir, "test_checkpoint")
        os.makedirs(checkpoint_dir)
        
        try:
            # Create original model
            print("\n" + "="*80)
            print("SETUP: Creating Original Model")
            print("="*80)
            
            model1 = DyLoRA_MoE(
                self.model_name,
                num_experts=self.num_experts,
                token=hf_token,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                allow_expert_growth=False,
                balance_coefficient=0.01
            )
            self.log("Created original model")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
            self.log("Loaded tokenizer")
            
            # Run tests
            test_results = {}
            
            # Test 1: Save checkpoint
            test_results["save"] = self.test_save_checkpoint(model1, checkpoint_dir)
            
            # Test 2: Verify structure
            test_results["structure"] = self.test_checkpoint_structure(checkpoint_dir)
            
            # Test 3: Load checkpoint
            model2 = self.test_load_checkpoint(checkpoint_dir, hf_token)
            test_results["load"] = model2 is not None
            
            # Test 4: Output consistency (if load succeeded)
            if model2 is not None:
                test_results["consistency"] = self.test_output_consistency(model1, model2, tokenizer)
            else:
                test_results["consistency"] = False
                self.log("Skipped consistency test (load failed)", False)
            
            # Test 5: Checkpoint size
            test_results["size"] = self.test_checkpoint_size(checkpoint_dir)
            
            # Summary
            print("\n" + "="*80)
            print("TEST SUMMARY")
            print("="*80)
            
            all_passed = all(test_results.values())
            for test_name, passed in test_results.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{status}: {test_name}")
            
            print("="*80)
            if all_passed:
                print("✓ ALL TESTS PASSED")
            else:
                print("✗ SOME TESTS FAILED")
            print("="*80)
            
            return {
                "all_passed": all_passed,
                "test_results": test_results,
                "checkpoint_dir": checkpoint_dir
            }
            
        except Exception as e:
            print(f"\n✗ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "all_passed": False,
                "error": str(e)
            }
        
        finally:
            # Cleanup
            if self.test_dir and os.path.exists(self.test_dir):
                print(f"\nℹ️  Test checkpoint saved at: {checkpoint_dir}")
                print(f"   (Run 'rm -rf {self.test_dir}' to clean up)")


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DyLoRA-MoE checkpoint save/load")
    parser.add_argument("--model", default="google/codegemma-2b", help="Base model name")
    parser.add_argument("--experts", type=int, default=2, help="Number of experts")
    parser.add_argument("--hf_token", help="HuggingFace token (optional)")
    parser.add_argument("--keep_checkpoint", action="store_true", 
                       help="Keep test checkpoint after tests")
    
    args = parser.parse_args()
    
    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Run tests
    tester = CheckpointTester(model_name=args.model, num_experts=args.experts)
    results = tester.run_all_tests(hf_token=hf_token)
    
    # Cleanup if requested
    if not args.keep_checkpoint and tester.test_dir:
        print(f"\nCleaning up test directory: {tester.test_dir}")
        shutil.rmtree(tester.test_dir)
    
    # Exit with appropriate code
    sys.exit(0 if results.get("all_passed", False) else 1)


if __name__ == "__main__":
    main()
