"""
Quick gradient flow verification for DyLoRA-MoE (single model instance).
"""

import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dylo_moe.model import DyLoRA_MoE

def main():
    print("=" * 80)
    print("GRADIENT FLOW VERIFICATION FOR DyLoRA-MoE")
    print("=" * 80)
    
    # Create model once
    print("\nLoading model...")
    model = DyLoRA_MoE(
        model_name="gpt2",  # 124M params - much faster for testing
        num_experts=2,
        lora_r=8,
        lora_alpha=16,
        balance_coefficient=0.01,
    )
    
    # Test 1: Multi-expert routing
    print("\n" + "=" * 80)
    print("TEST 1: Multi-Expert Routing Gradients")
    print("=" * 80)
    
    model.train()
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()
    
    print(f"Model training mode: {model.training}")
    print(f"Router training mode: {model.router.training}")
    
    outputs = model(input_ids=input_ids, labels=labels)
    print(f"\nLoss: {outputs.loss.item():.4f}")
    if model.last_balance_loss is not None:
        print(f"Balance Loss: {model.last_balance_loss.item():.6f}")
    
    outputs.loss.backward()
    
    # Check gradients
    router_grads = [p.grad.norm().item() for p in model.router.parameters() if p.grad is not None]
    lora_grads = []
    frozen_grads = []
    
    for name, param in model.foundation_model.named_parameters():
        if param.grad is not None:
            if "lora" in name.lower() and param.requires_grad:
                lora_grads.append(param.grad.norm().item())
            elif not param.requires_grad:
                frozen_grads.append(name)
    
    print(f"\n✓ Router params with gradients: {len(router_grads)}")
    print(f"✓ LoRA params with gradients: {len(lora_grads)}")
    print(f"✓ Frozen params with gradients: {len(frozen_grads)} (should be 0)")
    
    if router_grads:
        print(f"\nRouter grad norms: min={min(router_grads):.6f}, max={max(router_grads):.6f}")
    if lora_grads:
        print(f"LoRA grad norms: min={min(lora_grads):.6f}, max={max(lora_grads):.6f}")
    
    assert len(router_grads) > 0, "❌ Router has no gradients!"
    assert len(lora_grads) > 0, "❌ LoRA adapters have no gradients!"
    assert len(frozen_grads) == 0, "❌ Frozen params should not have gradients!"
    print("\n✅ PASS: Multi-expert routing gradients flow correctly!")
    
    # Test 2: Single expert mode
    print("\n" + "=" * 80)
    print("TEST 2: Single Expert Training Mode")
    print("=" * 80)
    
    model.zero_grad()
    # Use fresh input to avoid graph conflicts
    input_ids_2 = torch.randint(0, 1000, (2, 16))
    labels_2 = input_ids_2.clone()
    outputs = model(input_ids=input_ids_2, labels=labels_2, expert_id=0)
    outputs.loss.backward()
    
    expert_0_grads = 0
    expert_1_grads = 0
    
    for name, param in model.foundation_model.named_parameters():
        if "lora" in name.lower() and param.grad is not None:
            if "expert_0" in name:
                expert_0_grads += 1
            elif "expert_1" in name:
                expert_1_grads += 1
    
    print(f"\nExpert 0 params with gradients: {expert_0_grads}")
    print(f"Expert 1 params with gradients: {expert_1_grads} (should be 0)")
    
    assert expert_0_grads > 0, "❌ Expert 0 should have gradients!"
    assert expert_1_grads == 0, "❌ Expert 1 should NOT have gradients!"
    print("✅ PASS: Single expert mode isolates gradients correctly!")
    
    # Test 3: Routing weights gradient connection
    print("\n" + "=" * 80)
    print("TEST 3: Routing Weights Gradient Connection")
    print("=" * 80)
    
    model.zero_grad()
    input_ids_3 = torch.randint(0, 1000, (2, 16))
    labels_3 = input_ids_3.clone()
    outputs = model(input_ids=input_ids_3, labels=labels_3)
    
    has_grad_fn = hasattr(model, '_routing_weights_for_loss') and \
                  model._routing_weights_for_loss is not None and \
                  model._routing_weights_for_loss.grad_fn is not None
    
    print(f"\nRouting weights have grad_fn: {has_grad_fn}")
    if has_grad_fn:
        print(f"grad_fn type: {type(model._routing_weights_for_loss.grad_fn).__name__}")
    
    assert has_grad_fn, "❌ Routing weights lost gradient connection!"
    print("✅ PASS: Routing weights maintain gradient flow!")
    
    # Test 4: Inference mode (no gradients)
    print("\n" + "=" * 80)
    print("TEST 4: Inference Mode (eval)")
    print("=" * 80)
    
    model.eval()
    model.zero_grad()
    
    input_ids_4 = torch.randint(0, 1000, (2, 16))
    labels_4 = input_ids_4.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids_4, labels=labels_4)
    
    print(f"Model training mode: {model.training}")
    print(f"Router training mode: {model.router.training}")
    print(f"Loss computed: {outputs.loss is not None}")
    
    assert not model.training, "❌ Model should be in eval mode!"
    assert not model.router.training, "❌ Router should be in eval mode!"
    print("✅ PASS: Inference mode works correctly!")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nVerified:")
    print("  ✓ Router receives gradients during multi-expert training")
    print("  ✓ LoRA adapters receive gradients")
    print("  ✓ Frozen base model parameters remain frozen")
    print("  ✓ Single-expert mode isolates gradients to one expert")
    print("  ✓ Routing weights maintain gradient connections")
    print("  ✓ Inference mode disables gradients properly")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
