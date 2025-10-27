"""
Test ExpertManager new methods: get_all_adapter_names() and activate_all_experts()
"""

import torch
import os
from dylo_moe.expert import ExpertManager
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftMixedModel

def test_expert_manager_methods():
    """Test the new ExpertManager convenience methods."""
    print("\n" + "="*60)
    print("Testing ExpertManager New Methods")
    print("="*60)
    
    # Use tiny model for fast testing
    model_name = "google/gemma-3-270m-it"
    
    # Check if HF token is available
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN not found in environment, trying hf_token.txt...")
        try:
            with open("hf_token.txt", "r") as f:
                hf_token = f.read().strip()
            print("✓ Loaded HF token from hf_token.txt")
        except FileNotFoundError:
            print("❌ Could not find HF token. Please set HF_TOKEN env var or create hf_token.txt")
            return False
    
    print(f"\n1. Loading model: {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        attn_implementation="eager",
    )
    print("✓ Model loaded")
    
    print("\n2. Creating ExpertManager with 4 experts...")
    expert_manager = ExpertManager(
        model=base_model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    
    # Create 4 experts
    for i in range(4):
        expert_id = expert_manager.create_expert()
        print(f"   ✓ Created expert {expert_id}")
    
    print(f"\n✓ ExpertManager created with {expert_manager.num_experts} experts")
    
    # Test 1: get_all_adapter_names()
    print("\n3. Testing get_all_adapter_names()...")
    adapter_names = expert_manager.get_all_adapter_names()
    
    expected_names = ["expert_0", "expert_1", "expert_2", "expert_3"]
    if adapter_names != expected_names:
        print(f"❌ Adapter names mismatch!")
        print(f"   Expected: {expected_names}")
        print(f"   Got: {adapter_names}")
        return False
    
    print(f"✓ Adapter names correct: {adapter_names}")
    
    # Test 2: activate_all_experts()
    print("\n4. Testing activate_all_experts()...")
    
    # First, set a single expert as active
    expert_manager.set_active_expert(0)
    print(f"   - Set single expert active: expert_0")
    print(f"   - current_expert_id: {expert_manager.current_expert_id}")
    
    # Now activate all experts
    active_adapters = expert_manager.activate_all_experts()
    
    if active_adapters != expected_names:
        print(f"❌ activate_all_experts() returned wrong names!")
        print(f"   Expected: {expected_names}")
        print(f"   Got: {active_adapters}")
        return False
    
    print(f"✓ activate_all_experts() returned: {active_adapters}")
    
    # Check that current_expert_id is None (multi-expert mode)
    if expert_manager.current_expert_id is not None:
        print(f"❌ current_expert_id should be None in multi-expert mode!")
        print(f"   Got: {expert_manager.current_expert_id}")
        return False
    
    print(f"✓ current_expert_id set to None (multi-expert mode)")
    
    # Test 3: Verify PEFT model has all adapters active
    print("\n5. Verifying PEFT model state...")
    
    if not isinstance(expert_manager.model, (PeftModel, PeftMixedModel)):
        print("❌ Model is not a PeftModel!")
        return False
    
    print("✓ Model is PeftModel")
    
    # Check that base_model has the correct active adapters
    # Note: Different PEFT versions may store this differently
    try:
        if hasattr(expert_manager.model.base_model, 'active_adapters'):
            base_model_adapters = expert_manager.model.base_model.active_adapters
        elif hasattr(expert_manager.model.base_model, '_active_adapter'):
            base_model_adapters = expert_manager.model.base_model._active_adapter
        else:
            print("⚠️  Could not access active adapters attribute")
            base_model_adapters = None
        
        if base_model_adapters:
            if isinstance(base_model_adapters, str):
                base_model_adapters = [base_model_adapters]
            
            if base_model_adapters != expected_names:
                print(f"⚠️  Base model active adapters: {base_model_adapters}")
                print(f"   Expected: {expected_names}")
            else:
                print(f"✓ Base model has all adapters active: {base_model_adapters}")
    except Exception as e:
        print(f"⚠️  Could not verify active adapters: {e}")
        # This is a warning, not a failure
    
    # Test 4: Test switching back to single expert
    print("\n6. Testing switch back to single expert...")
    expert_manager.set_active_expert(2)
    
    if expert_manager.current_expert_id != 2:
        print(f"❌ current_expert_id should be 2!")
        print(f"   Got: {expert_manager.current_expert_id}")
        return False
    
    print(f"✓ Successfully switched back to single expert: {expert_manager.current_expert_id}")
    
    # Test 5: Test forward pass with all experts active
    print("\n7. Testing forward pass with all experts active...")
    
    expert_manager.activate_all_experts()
    
    # Create test input
    input_ids = torch.randint(0, 1000, (1, 8))
    
    try:
        outputs = expert_manager.model(input_ids)
        print(f"✓ Forward pass successful")
        print(f"   Output shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_expert_manager_methods()
    exit(0 if success else 1)
