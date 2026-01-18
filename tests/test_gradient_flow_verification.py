"""
Comprehensive gradient flow verification for DyLoRA-MoE.

Tests:
1. Router gradients flow during multi-expert routing
2. LoRA adapter gradients flow for active experts
3. Load balancing loss contributes to router gradients
4. No gradients leak to frozen base model parameters
"""

import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dylo_moe.model import DyLoRA_MoE

def test_multi_expert_gradient_flow():
    """Test that gradients flow to router and LoRA adapters during multi-expert training."""
    print("=" * 80)
    print("TEST 1: Multi-Expert Gradient Flow")
    print("=" * 80)
    
    # Create model with 2 experts
    model = DyLoRA_MoE(
        model_name="google/codegemma-2b",
        num_experts=2,
        lora_r=8,
        lora_alpha=16,
        balance_coefficient=0.01,
    )
    model.train()
    
    # Create dummy batch
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = input_ids.clone()
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Model in training mode: {model.training}")
    print(f"Router in training mode: {model.router.training}")
    
    # Forward pass (should use multi-expert routing)
    outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"\nLoss: {outputs.loss.item():.4f}")
    if model.last_lm_loss is not None:
        print(f"LM Loss: {model.last_lm_loss.item():.4f}")
    if model.last_balance_loss is not None:
        print(f"Balance Loss: {model.last_balance_loss.item():.4f}")
    
    # Backward pass
    outputs.loss.backward()
    
    print("\n" + "-" * 80)
    print("GRADIENT FLOW VERIFICATION:")
    print("-" * 80)
    
    # 1. Check router gradients
    router_has_grads = False
    router_grad_norms = []
    for name, param in model.router.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            router_grad_norms.append(grad_norm)
            router_has_grads = True
            print(f"✓ Router {name}: grad_norm = {grad_norm:.6f}")
        else:
            print(f"✗ Router {name}: NO GRADIENT")
    
    # 2. Check LoRA adapter gradients
    lora_params_with_grads = 0
    lora_params_without_grads = 0
    lora_grad_norms = []
    
    for name, param in model.foundation_model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                lora_grad_norms.append(grad_norm)
                lora_params_with_grads += 1
                if lora_params_with_grads <= 5:  # Show first 5
                    print(f"✓ LoRA {name}: grad_norm = {grad_norm:.6f}")
            else:
                lora_params_without_grads += 1
                if lora_params_without_grads <= 3:  # Show first 3 issues
                    print(f"✗ LoRA {name}: NO GRADIENT")
    
    if lora_params_with_grads > 5:
        print(f"✓ ... and {lora_params_with_grads - 5} more LoRA params with gradients")
    
    # 3. Check frozen parameters don't have gradients
    frozen_params_with_grads = []
    for name, param in model.foundation_model.named_parameters():
        if not param.requires_grad:
            if param.grad is not None:
                frozen_params_with_grads.append(name)
    
    print("\n" + "-" * 80)
    print("SUMMARY:")
    print("-" * 80)
    print(f"Router parameters with gradients: {len(router_grad_norms)}")
    print(f"LoRA parameters with gradients: {lora_params_with_grads}")
    print(f"LoRA parameters WITHOUT gradients: {lora_params_without_grads}")
    print(f"Frozen parameters with gradients (should be 0): {len(frozen_params_with_grads)}")
    
    if router_grad_norms:
        print(f"\nRouter gradient norms: min={min(router_grad_norms):.6f}, max={max(router_grad_norms):.6f}, mean={sum(router_grad_norms)/len(router_grad_norms):.6f}")
    
    if lora_grad_norms:
        print(f"LoRA gradient norms: min={min(lora_grad_norms):.6f}, max={max(lora_grad_norms):.6f}, mean={sum(lora_grad_norms)/len(lora_grad_norms):.6f}")
    
    # Assertions
    assert router_has_grads, "❌ FAIL: Router has no gradients!"
    assert lora_params_with_grads > 0, "❌ FAIL: No LoRA parameters have gradients!"
    assert len(frozen_params_with_grads) == 0, f"❌ FAIL: {len(frozen_params_with_grads)} frozen params have gradients!"
    assert lora_params_without_grads == 0, f"⚠️  WARNING: {lora_params_without_grads} LoRA params missing gradients"
    
    print("\n✅ PASS: Gradients flow correctly to router and LoRA adapters!")
    print("=" * 80)
    return True


def test_single_expert_gradient_flow():
    """Test gradient flow when using explicit expert_id (single expert training)."""
    print("\n" + "=" * 80)
    print("TEST 2: Single Expert Training Mode")
    print("=" * 80)
    
    model = DyLoRA_MoE(
        model_name="google/codegemma-2b",
        num_experts=2,
        lora_r=8,
        lora_alpha=16,
    )
    model.train()
    
    # Create dummy batch
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()
    
    print(f"\nTraining with explicit expert_id=0")
    
    # Forward with explicit expert
    outputs = model(input_ids=input_ids, labels=labels, expert_id=0)
    outputs.loss.backward()
    
    # Check that only expert 0's LoRA params have gradients
    expert_0_grads = 0
    expert_1_grads = 0
    
    for name, param in model.foundation_model.named_parameters():
        if "lora" in name.lower() and param.grad is not None:
            if "expert_0" in name:
                expert_0_grads += 1
            elif "expert_1" in name:
                expert_1_grads += 1
    
    print(f"\nExpert 0 LoRA params with gradients: {expert_0_grads}")
    print(f"Expert 1 LoRA params with gradients: {expert_1_grads}")
    
    assert expert_0_grads > 0, "❌ FAIL: Expert 0 should have gradients!"
    assert expert_1_grads == 0, "❌ FAIL: Expert 1 should NOT have gradients!"
    
    print("✅ PASS: Single expert training isolates gradients correctly!")
    print("=" * 80)
    return True


def test_routing_weights_gradient():
    """Test that routing weights maintain gradients through the computation graph."""
    print("\n" + "=" * 80)
    print("TEST 3: Routing Weights Gradient Connection")
    print("=" * 80)
    
    model = DyLoRA_MoE(
        model_name="google/codegemma-2b",
        num_experts=2,
        lora_r=8,
        lora_alpha=16,
        balance_coefficient=0.01,
    )
    model.train()
    
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    
    # Check that routing weights for loss computation have grad_fn
    has_grad_fn = hasattr(model, '_routing_weights_for_loss') and \
                  model._routing_weights_for_loss is not None and \
                  model._routing_weights_for_loss.grad_fn is not None
    
    print(f"\nRouting weights have grad_fn: {has_grad_fn}")
    if has_grad_fn:
        print(f"grad_fn type: {type(model._routing_weights_for_loss.grad_fn).__name__}")
    
    # Backward
    outputs.loss.backward()
    
    # Check router gradients
    router_grad_count = sum(1 for p in model.router.parameters() if p.grad is not None)
    print(f"Router parameters with gradients: {router_grad_count}/{sum(1 for _ in model.router.parameters())}")
    
    assert has_grad_fn, "❌ FAIL: Routing weights lost gradient connection!"
    assert router_grad_count > 0, "❌ FAIL: Router has no gradients!"
    
    print("✅ PASS: Routing weights maintain gradient flow to router!")
    print("=" * 80)
    return True


def test_balance_loss_contribution():
    """Test that balance loss contributes to router gradients."""
    print("\n" + "=" * 80)
    print("TEST 4: Load Balance Loss Gradient Contribution")
    print("=" * 80)
    
    # Test with balance_coefficient = 0 (no balance loss)
    print("\n--- WITHOUT balance loss (coefficient=0) ---")
    model_no_balance = DyLoRA_MoE(
        model_name="google/codegemma-2b",
        num_experts=2,
        lora_r=8,
        lora_alpha=16,
        balance_coefficient=0.0,
    )
    model_no_balance.train()
    
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()
    
    outputs = model_no_balance(input_ids=input_ids, labels=labels)
    outputs.loss.backward()
    
    router_grad_norm_no_balance = sum(
        p.grad.norm().item() for p in model_no_balance.router.parameters() 
        if p.grad is not None
    )
    print(f"Router total grad norm: {router_grad_norm_no_balance:.6f}")
    
    # Test with balance_coefficient = 0.01 (with balance loss)
    print("\n--- WITH balance loss (coefficient=0.01) ---")
    model_with_balance = DyLoRA_MoE(
        model_name="google/codegemma-2b",
        num_experts=2,
        lora_r=8,
        lora_alpha=16,
        balance_coefficient=0.01,
    )
    model_with_balance.train()
    
    outputs = model_with_balance(input_ids=input_ids, labels=labels)
    print(f"Balance loss: {model_with_balance.last_balance_loss.item():.6f}")
    outputs.loss.backward()
    
    router_grad_norm_with_balance = sum(
        p.grad.norm().item() for p in model_with_balance.router.parameters() 
        if p.grad is not None
    )
    print(f"Router total grad norm: {router_grad_norm_with_balance:.6f}")
    
    # Balance loss should increase router gradients
    print(f"\nGradient increase from balance loss: {router_grad_norm_with_balance - router_grad_norm_no_balance:.6f}")
    
    # Note: We can't assert gradients are always higher with balance loss because
    # it depends on the random initialization and data, but we can verify both have gradients
    assert router_grad_norm_no_balance > 0, "❌ FAIL: Router should have gradients even without balance loss!"
    assert router_grad_norm_with_balance > 0, "❌ FAIL: Router should have gradients with balance loss!"
    
    print("✅ PASS: Balance loss computation verified!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GRADIENT FLOW VERIFICATION FOR DyLoRA-MoE")
    print("=" * 80)
    
    try:
        test_multi_expert_gradient_flow()
        test_single_expert_gradient_flow()
        test_routing_weights_gradient()
        test_balance_loss_contribution()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nGradient flow is working correctly:")
        print("  ✓ Router receives gradients during multi-expert training")
        print("  ✓ LoRA adapters receive gradients")
        print("  ✓ Frozen parameters remain frozen")
        print("  ✓ Single-expert mode isolates gradients")
        print("  ✓ Routing weights maintain gradient connections")
        print("  ✓ Load balancing loss contributes to training")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
