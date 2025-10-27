#!/usr/bin/env python3
"""
Test suite for MoELoraConfig
"""

import sys
from peft import MoELoraConfig, LoraConfig


def test_moe_config_creation():
    """Test basic MoELoraConfig instantiation"""
    print("\n" + "=" * 60)
    print("Test 1: Basic MoELoraConfig Creation")
    print("=" * 60)
    
    config = MoELoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        use_moe_routing=True,
        num_experts=4,
        top_k_experts=2,
        router_hidden_size=2560,
        router_aux_loss_coef=0.01,
    )
    
    print(f"✓ Config created successfully")
    print(f"  r: {config.r}")
    print(f"  lora_alpha: {config.lora_alpha}")
    print(f"  use_moe_routing: {config.use_moe_routing}")
    print(f"  num_experts: {config.num_experts}")
    print(f"  top_k_experts: {config.top_k_experts}")
    print(f"  router_hidden_size: {config.router_hidden_size}")
    print(f"  router_aux_loss_coef: {config.router_aux_loss_coef}")
    print(f"  router_temperature: {config.router_temperature}")
    print(f"  router_type: {config.router_type}")
    print(f"  expert_capacity_factor: {config.expert_capacity_factor}")
    print(f"  expert_dropout: {config.expert_dropout}")
    print(f"  load_balance_loss_type: {config.load_balance_loss_type}")
    
    assert config.use_moe_routing is True
    assert config.num_experts == 4
    assert config.top_k_experts == 2
    assert config.router_hidden_size == 2560
    
    print("\n✅ Test 1 PASSED: Basic config creation works!")
    return True


def test_inheritance():
    """Test that MoELoraConfig inherits from LoraConfig"""
    print("\n" + "=" * 60)
    print("Test 2: MoELoraConfig Inheritance")
    print("=" * 60)
    
    config = MoELoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        num_experts=3,
    )
    
    assert isinstance(config, LoraConfig)
    print(f"✓ MoELoraConfig is instance of LoraConfig")
    
    # Check inherited attributes
    assert config.r == 8
    assert config.lora_alpha == 16
    # target_modules is converted to set by parent's __post_init__
    assert config.target_modules == {"c_attn"}
    print(f"✓ Inherited LoRA attributes accessible")
    
    # Check MoE-specific attributes
    assert config.num_experts == 3
    assert config.use_moe_routing is True
    print(f"✓ MoE-specific attributes accessible")
    
    print("\n✅ Test 2 PASSED: Inheritance works correctly!")
    return True


def test_validation_num_experts():
    """Test validation of num_experts"""
    print("\n" + "=" * 60)
    print("Test 3: Validation - num_experts")
    print("=" * 60)
    
    # Should raise error for num_experts < 2
    try:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            use_moe_routing=True,
            num_experts=1,  # Invalid
        )
        print("❌ Should have raised ValueError for num_experts=1")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Valid num_experts should work
    config = MoELoraConfig(
        r=8,
        target_modules=["q_proj"],
        num_experts=2,
    )
    print(f"✓ num_experts=2 works correctly")
    
    print("\n✅ Test 3 PASSED: num_experts validation works!")
    return True


def test_validation_top_k():
    """Test validation of top_k_experts"""
    print("\n" + "=" * 60)
    print("Test 4: Validation - top_k_experts")
    print("=" * 60)
    
    # Should raise error for top_k > num_experts
    try:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            num_experts=3,
            top_k_experts=4,  # Invalid: > num_experts
        )
        print("❌ Should have raised ValueError for top_k > num_experts")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Should raise error for top_k < 1
    try:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            num_experts=3,
            top_k_experts=0,  # Invalid: < 1
        )
        print("❌ Should have raised ValueError for top_k=0")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Valid top_k should work
    config = MoELoraConfig(
        r=8,
        target_modules=["q_proj"],
        num_experts=4,
        top_k_experts=2,
    )
    print(f"✓ top_k_experts=2 with num_experts=4 works correctly")
    
    print("\n✅ Test 4 PASSED: top_k_experts validation works!")
    return True


def test_validation_temperature():
    """Test validation of router_temperature"""
    print("\n" + "=" * 60)
    print("Test 5: Validation - router_temperature")
    print("=" * 60)
    
    # Should raise error for temperature <= 0
    try:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            router_temperature=0.0,  # Invalid
        )
        print("❌ Should have raised ValueError for router_temperature=0")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    try:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            router_temperature=-1.0,  # Invalid
        )
        print("❌ Should have raised ValueError for router_temperature=-1")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Valid temperature should work
    config = MoELoraConfig(
        r=8,
        target_modules=["q_proj"],
        router_temperature=2.0,
    )
    print(f"✓ router_temperature=2.0 works correctly")
    
    print("\n✅ Test 5 PASSED: router_temperature validation works!")
    return True


def test_serialization():
    """Test config serialization (to_dict, from_dict)"""
    print("\n" + "=" * 60)
    print("Test 6: Config Serialization")
    print("=" * 60)
    
    # Create config
    config = MoELoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        num_experts=4,
        top_k_experts=2,
        router_aux_loss_coef=0.05,
        router_temperature=1.5,
    )
    
    # Convert to dict
    config_dict = config.to_dict()
    print(f"✓ to_dict() successful")
    print(f"  Keys: {list(config_dict.keys())[:10]}...")  # Show first 10 keys
    
    # Check MoE fields are in dict
    assert "num_experts" in config_dict
    assert "top_k_experts" in config_dict
    assert "use_moe_routing" in config_dict
    assert config_dict["num_experts"] == 4
    assert config_dict["top_k_experts"] == 2
    print(f"✓ MoE fields present in serialized dict")
    
    # Check inherited fields
    assert "r" in config_dict
    assert "lora_alpha" in config_dict
    assert config_dict["r"] == 16
    print(f"✓ Inherited LoRA fields present in serialized dict")
    
    print("\n✅ Test 6 PASSED: Serialization works correctly!")
    return True


def test_default_values():
    """Test default values for MoE parameters"""
    print("\n" + "=" * 60)
    print("Test 7: Default Values")
    print("=" * 60)
    
    config = MoELoraConfig(
        r=8,
        target_modules=["q_proj"],
    )
    
    # Check defaults
    assert config.use_moe_routing is True
    print(f"✓ use_moe_routing defaults to True")
    
    assert config.num_experts == 4
    print(f"✓ num_experts defaults to 4")
    
    assert config.top_k_experts == 2
    print(f"✓ top_k_experts defaults to 2")
    
    assert config.router_hidden_size is None
    print(f"✓ router_hidden_size defaults to None")
    
    assert config.router_aux_loss_coef == 0.01
    print(f"✓ router_aux_loss_coef defaults to 0.01")
    
    assert config.router_temperature == 2.0
    print(f"✓ router_temperature defaults to 2.0")
    
    assert config.router_type == "learned_with_maturity"
    print(f"✓ router_type defaults to 'learned_with_maturity'")
    
    assert config.expert_capacity_factor == 1.25
    print(f"✓ expert_capacity_factor defaults to 1.25")
    
    assert config.expert_dropout == 0.0
    print(f"✓ expert_dropout defaults to 0.0")
    
    assert config.load_balance_loss_type == "aux_loss"
    print(f"✓ load_balance_loss_type defaults to 'aux_loss'")
    
    print("\n✅ Test 7 PASSED: Default values correct!")
    return True


def test_disabled_moe():
    """Test behavior when use_moe_routing=False"""
    print("\n" + "=" * 60)
    print("Test 8: Disabled MoE Routing")
    print("=" * 60)
    
    # Should work with use_moe_routing=False even with num_experts=1
    config = MoELoraConfig(
        r=8,
        target_modules=["q_proj"],
        use_moe_routing=False,
        num_experts=1,  # Would be invalid if use_moe_routing=True
    )
    
    print(f"✓ Config created with use_moe_routing=False, num_experts=1")
    assert config.use_moe_routing is False
    assert config.num_experts == 1
    
    print("\n✅ Test 8 PASSED: Disabled MoE works correctly!")
    return True


def test_router_types():
    """Test different router types"""
    print("\n" + "=" * 60)
    print("Test 9: Router Types")
    print("=" * 60)
    
    # Test valid router types
    for router_type in ["learned", "learned_with_maturity", "fixed"]:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            router_type=router_type,
        )
        print(f"✓ router_type='{router_type}' works")
    
    # Test invalid router type
    try:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            router_type="invalid_type",
        )
        print("❌ Should have raised ValueError for invalid router_type")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for invalid router_type: {e}")
    
    print("\n✅ Test 9 PASSED: Router types validation works!")
    return True


def test_load_balance_types():
    """Test different load balance loss types"""
    print("\n" + "=" * 60)
    print("Test 10: Load Balance Loss Types")
    print("=" * 60)
    
    # Test valid loss types
    for loss_type in ["aux_loss", "z_loss", "switch"]:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            load_balance_loss_type=loss_type,
        )
        print(f"✓ load_balance_loss_type='{loss_type}' works")
    
    # Test invalid loss type
    try:
        config = MoELoraConfig(
            r=8,
            target_modules=["q_proj"],
            load_balance_loss_type="invalid_type",
        )
        print("❌ Should have raised ValueError for invalid load_balance_loss_type")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for invalid load_balance_loss_type: {e}")
    
    print("\n✅ Test 10 PASSED: Load balance loss types validation works!")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing MoELoraConfig")
    print("=" * 60)
    
    tests = [
        test_moe_config_creation,
        test_inheritance,
        test_validation_num_experts,
        test_validation_top_k,
        test_validation_temperature,
        test_serialization,
        test_default_values,
        test_disabled_moe,
        test_router_types,
        test_load_balance_types,
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
    sys.exit(main())
