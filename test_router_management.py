#!/usr/bin/env python3
"""
Test suite for LoraModel router management functionality
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, MoELoraConfig
from transformers import AutoModelForCausalLM


class SimpleRouter(nn.Module):
    """Simple router for testing"""
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_experts)
        
    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_size]
        logits = self.linear(hidden_states)  # [batch, seq_len, num_experts]
        return torch.softmax(logits, dim=-1)


def test_attach_router():
    """Test attaching a router to LoraModel"""
    print("\n" + "=" * 60)
    print("Test 1: Attach Router")
    print("=" * 60)
    
    # Create model with LoRA
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    
    # Add more experts
    model.add_adapter("expert_1", lora_config)
    model.add_adapter("expert_2", lora_config)
    
    # Create router
    router = SimpleRouter(hidden_size=768, num_experts=3)
    
    # Create MoE config
    moe_config = MoELoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        num_experts=3,
    )
    
    # Attach router
    model.base_model.attach_router(router, moe_config)
    
    print(f"✓ Router attached successfully")
    assert model.base_model.has_router()
    print(f"✓ has_router() returns True")
    assert model.base_model.router is not None
    print(f"✓ router attribute is not None")
    assert model.base_model.moe_config is not None
    print(f"✓ moe_config attribute is not None")
    
    print("\n✅ Test 1 PASSED!")
    return True


def test_detach_router():
    """Test detaching a router from LoraModel"""
    print("\n" + "=" * 60)
    print("Test 2: Detach Router")
    print("=" * 60)
    
    # Create model with LoRA
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    
    # Create and attach router
    router = SimpleRouter(hidden_size=768, num_experts=2)
    model.base_model.attach_router(router)
    
    print(f"✓ Router attached")
    assert model.base_model.has_router()
    
    # Detach router
    model.base_model.detach_router()
    
    print(f"✓ Router detached successfully")
    assert not model.base_model.has_router()
    print(f"✓ has_router() returns False")
    assert model.base_model.router is None
    print(f"✓ router attribute is None")
    assert model.base_model.moe_config is None
    print(f"✓ moe_config attribute is None")
    
    print("\n✅ Test 2 PASSED!")
    return True


def test_get_routing_weights():
    """Test computing routing weights"""
    print("\n" + "=" * 60)
    print("Test 3: Get Routing Weights")
    print("=" * 60)
    
    # Create model with LoRA
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    model.add_adapter("expert_1", lora_config)
    model.add_adapter("expert_2", lora_config)
    model.add_adapter("expert_3", lora_config)
    
    # Create and attach router
    router = SimpleRouter(hidden_size=768, num_experts=4)
    model.base_model.attach_router(router)
    
    # Create hidden states
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Get routing weights
    routing_weights = model.base_model.get_routing_weights(hidden_states)
    
    print(f"✓ Routing weights computed")
    assert routing_weights is not None
    print(f"✓ Routing weights is not None")
    assert routing_weights.shape == (batch_size, seq_len, 4)
    print(f"✓ Shape is correct: {routing_weights.shape}")
    
    # Check that weights sum to 1 (softmax)
    weight_sums = routing_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    print(f"✓ Weights sum to 1 (softmax normalized)")
    
    print("\n✅ Test 3 PASSED!")
    return True


def test_get_routing_weights_no_router():
    """Test get_routing_weights returns None when no router attached"""
    print("\n" + "=" * 60)
    print("Test 4: Get Routing Weights Without Router")
    print("=" * 60)
    
    # Create model with LoRA
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    
    # Create hidden states
    hidden_states = torch.randn(2, 10, 768)
    
    # Get routing weights without router
    routing_weights = model.base_model.get_routing_weights(hidden_states)
    
    print(f"✓ get_routing_weights() called without router")
    assert routing_weights is None
    print(f"✓ Returns None when no router attached")
    
    print("\n✅ Test 4 PASSED!")
    return True


def test_forward_with_router():
    """Test full forward pass with router and routing_weights"""
    print("\n" + "=" * 60)
    print("Test 5: Forward Pass with Router")
    print("=" * 60)
    
    # Create model with LoRA
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    model.add_adapter("expert_1", lora_config)
    model.add_adapter("expert_2", lora_config)
    
    # Activate all experts
    model.base_model.set_adapter(["expert_0", "expert_1", "expert_2"])
    
    # Create and attach router
    router = SimpleRouter(hidden_size=768, num_experts=3)
    model.base_model.attach_router(router)
    
    # Create input
    input_ids = torch.randint(0, 1000, (2, 10))
    
    # Get hidden states (simplified - just use random for testing)
    hidden_states = torch.randn(2, 10, 768)
    
    # Get routing weights
    routing_weights = model.base_model.get_routing_weights(hidden_states)
    print(f"✓ Routing weights shape: {routing_weights.shape}")
    
    # Forward pass with routing_weights
    outputs = model(input_ids, routing_weights=routing_weights)
    
    print(f"✓ Forward pass successful")
    assert outputs.logits is not None
    print(f"✓ Output logits shape: {outputs.logits.shape}")
    assert outputs.logits.shape == (2, 10, 50257)  # GPT-2 vocab size
    print(f"✓ Output shape is correct")
    
    print("\n✅ Test 5 PASSED!")
    return True


def test_router_device_placement():
    """Test router moves to correct device with model"""
    print("\n" + "=" * 60)
    print("Test 6: Router Device Placement")
    print("=" * 60)
    
    # Create model with LoRA
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    
    # Create router
    router = SimpleRouter(hidden_size=768, num_experts=2)
    
    # Attach router
    model.base_model.attach_router(router)
    
    # Check router is on CPU
    router_device = next(model.base_model.router.parameters()).device
    print(f"✓ Router device after attach: {router_device}")
    assert router_device.type == "cpu"
    print(f"✓ Router is on CPU")
    
    # If CUDA available, test GPU placement
    if torch.cuda.is_available():
        model = model.to("cuda")
        router_device = next(model.base_model.router.parameters()).device
        print(f"✓ Router device after model.to('cuda'): {router_device}")
        assert router_device.type == "cuda"
        print(f"✓ Router moved to CUDA with model")
    else:
        print(f"⚠ CUDA not available, skipping GPU test")
    
    print("\n✅ Test 6 PASSED!")
    return True


def test_invalid_router_type():
    """Test error handling for invalid router type"""
    print("\n" + "=" * 60)
    print("Test 7: Invalid Router Type")
    print("=" * 60)
    
    # Create model with LoRA
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    
    # Try to attach invalid router (not nn.Module)
    try:
        model.base_model.attach_router("not_a_module")
        print("❌ Should have raised TypeError")
        return False
    except TypeError as e:
        print(f"✓ Correctly raised TypeError: {e}")
    
    print("\n✅ Test 7 PASSED!")
    return True


def test_invalid_moe_config_type():
    """Test error handling for invalid MoE config type"""
    print("\n" + "=" * 60)
    print("Test 8: Invalid MoE Config Type")
    print("=" * 60)
    
    # Create model with LoRA
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name="expert_0")
    
    # Create valid router
    router = SimpleRouter(hidden_size=768, num_experts=2)
    
    # Try to attach with invalid config (not MoELoraConfig)
    try:
        model.base_model.attach_router(router, moe_config="not_a_config")
        print("❌ Should have raised TypeError")
        return False
    except TypeError as e:
        print(f"✓ Correctly raised TypeError: {e}")
    
    print("\n✅ Test 8 PASSED!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing LoraModel Router Management")
    print("=" * 60)
    
    tests = [
        test_attach_router,
        test_detach_router,
        test_get_routing_weights,
        test_get_routing_weights_no_router,
        test_forward_with_router,
        test_router_device_placement,
        test_invalid_router_type,
        test_invalid_moe_config_type,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{len(tests)} passed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
