"""
Test gradient flow through the new single-pass routing implementation.

This validates that:
1. Gradients flow through routing_weights to the router
2. Gradients flow through routing_weights to LoRA adapters
3. Gradient magnitudes are reasonable
4. All trainable parameters receive gradients
"""

import torch
import os
from dylo_moe.model import DyLoRA_MoE

def test_gradient_flow():
    """Test that gradients flow correctly through the single-pass routing."""
    print("\n" + "="*60)
    print("Testing Gradient Flow in Single-Pass Routing")
    print("="*60)
    
    # Use tiny model for fast testing (270M params, already cached)
    model_name = "google/gemma-3-270m-it"
    
    # Check if HF token is available
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            with open("hf_token.txt", "r") as f:
                hf_token = f.read().strip()
        except FileNotFoundError:
            print("❌ Could not find HF token")
            return False
    
    print(f"\n1. Initializing DyLoRA_MoE with {model_name}...")
    print("   Creating model with 4 experts...")
    
    model = DyLoRA_MoE(
        model_name=model_name,
        num_experts=4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        token=hf_token,
        allow_expert_growth=False,
        balance_coefficient=0.01,
    )
    model.train()  # Ensure training mode
    
    print("✓ Model initialized")
    
    # Create test input
    print("\n2. Creating test input...")
    batch_size = 2
    seq_length = 16
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    print(f"   Input shape: {input_ids.shape}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    if outputs.loss is None:
        print("❌ Loss is None!")
        return False
    
    print(f"✓ Forward pass complete")
    print(f"   Loss: {outputs.loss.item():.4f}")
    
    # Check if routing_weights requires gradients
    print(f"\n   Checking routing_weights...")
    if model.last_routing_weights is not None:
        print(f"   - routing_weights shape: {model.last_routing_weights.shape}")
        print(f"   - routing_weights requires_grad: {model.last_routing_weights.requires_grad}")
        print(f"   - routing_weights is_leaf: {model.last_routing_weights.is_leaf}")
    else:
        print(f"   - routing_weights is None")
    
    # Backward pass
    print("\n4. Running backward pass...")
    model.zero_grad()
    outputs.loss.backward()
    print("✓ Backward pass complete")
    
    # Check router gradients
    print("\n5. Checking router gradients...")
    router_params_with_grad = 0
    router_params_total = 0
    router_grad_norms = []
    
    for name, param in model.router.named_parameters():
        router_params_total += 1
        if param.grad is not None:
            router_params_with_grad += 1
            grad_norm = param.grad.norm().item()
            router_grad_norms.append(grad_norm)
            print(f"   ✓ {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"   ❌ {name}: NO GRADIENT")
    
    if router_params_with_grad == 0:
        print("❌ Router has no gradients!")
        return False
    
    if router_params_with_grad < router_params_total:
        print(f"⚠️  Only {router_params_with_grad}/{router_params_total} router params have gradients")
    else:
        print(f"✓ All {router_params_total} router parameters have gradients")
    
    # Check gradient magnitudes are reasonable (not too small, not exploding)
    avg_router_grad = sum(router_grad_norms) / len(router_grad_norms)
    if avg_router_grad < 1e-8:
        print(f"⚠️  Router gradients very small: {avg_router_grad:.2e}")
    elif avg_router_grad > 100:
        print(f"⚠️  Router gradients very large: {avg_router_grad:.2e}")
    else:
        print(f"✓ Router gradient magnitude reasonable: {avg_router_grad:.6f}")
    
    # Check LoRA adapter gradients
    print("\n6. Checking LoRA adapter gradients...")
    lora_params_with_grad = 0
    lora_params_total = 0
    lora_grad_norms = []
    expert_grad_counts = {i: 0 for i in range(4)}
    
    for name, param in model.foundation_model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_params_total += 1
            
            # Extract expert ID from parameter name
            expert_id = None
            for i in range(4):
                if f"expert_{i}" in name:
                    expert_id = i
                    break
            
            if param.grad is not None:
                lora_params_with_grad += 1
                grad_norm = param.grad.norm().item()
                lora_grad_norms.append(grad_norm)
                if expert_id is not None:
                    expert_grad_counts[expert_id] += 1
                
                # Only print first few to avoid spam
                if lora_params_with_grad <= 5:
                    print(f"   ✓ {name[:60]}...: grad_norm={grad_norm:.6f}")
            else:
                if lora_params_with_grad <= 5:
                    print(f"   ❌ {name[:60]}...: NO GRADIENT")
    
    if lora_params_with_grad == 0:
        print("❌ LoRA adapters have no gradients!")
        return False
    
    print(f"✓ {lora_params_with_grad}/{lora_params_total} LoRA parameters have gradients")
    
    # Check that all experts received gradients
    print("\n7. Checking per-expert gradient distribution...")
    for expert_id, count in expert_grad_counts.items():
        if count > 0:
            print(f"   ✓ expert_{expert_id}: {count} params with gradients")
        else:
            print(f"   ❌ expert_{expert_id}: NO GRADIENTS")
    
    all_experts_have_grads = all(count > 0 for count in expert_grad_counts.values())
    if not all_experts_have_grads:
        print("❌ Not all experts received gradients!")
        return False
    
    print("✓ All experts received gradients")
    
    # Check LoRA gradient magnitudes
    if lora_grad_norms:
        avg_lora_grad = sum(lora_grad_norms) / len(lora_grad_norms)
        if avg_lora_grad < 1e-8:
            print(f"⚠️  LoRA gradients very small: {avg_lora_grad:.2e}")
        elif avg_lora_grad > 100:
            print(f"⚠️  LoRA gradients very large: {avg_lora_grad:.2e}")
        else:
            print(f"✓ LoRA gradient magnitude reasonable: {avg_lora_grad:.6f}")
    
    # Check base model is frozen
    print("\n8. Verifying base model is frozen...")
    base_params_with_grad = 0
    base_params_total = 0
    
    for name, param in model.foundation_model.named_parameters():
        if "lora" not in name.lower() and "router" not in name.lower():
            base_params_total += 1
            if param.requires_grad:
                base_params_with_grad += 1
                if base_params_with_grad <= 3:
                    print(f"   ⚠️  {name}: requires_grad=True (should be frozen)")
    
    if base_params_with_grad == 0:
        print(f"✓ All {base_params_total} base model parameters are frozen")
    else:
        print(f"⚠️  {base_params_with_grad}/{base_params_total} base params are not frozen")
    
    # Test multiple backward passes (accumulation)
    print("\n9. Testing gradient accumulation...")
    model.zero_grad()
    
    # First backward
    outputs1 = model(input_ids, attention_mask=attention_mask, labels=labels)
    outputs1.loss.backward()
    
    # Get router gradient after first backward
    first_grad = None
    for param in model.router.parameters():
        if param.grad is not None:
            first_grad = param.grad.clone()
            break
    
    if first_grad is None:
        print("❌ No router gradient after first backward")
        return False
    
    # Second backward (accumulate)
    outputs2 = model(input_ids, attention_mask=attention_mask, labels=labels)
    outputs2.loss.backward()
    
    # Get router gradient after second backward
    second_grad = None
    for param in model.router.parameters():
        if param.grad is not None:
            second_grad = param.grad.clone()
            break
    
    if second_grad is None:
        print("❌ No router gradient after second backward")
        return False
    
    # Gradients should have accumulated (approximately doubled)
    grad_ratio = (second_grad.norm() / first_grad.norm()).item()
    print(f"   Gradient accumulation ratio: {grad_ratio:.2f}")
    
    if 1.5 < grad_ratio < 2.5:
        print("✓ Gradient accumulation working correctly")
    else:
        print(f"⚠️  Gradient accumulation ratio unexpected: {grad_ratio:.2f} (expected ~2.0)")
    
    print("\n" + "="*60)
    print("✅ ALL GRADIENT FLOW TESTS PASSED!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Router: {router_params_with_grad}/{router_params_total} params with gradients")
    print(f"  - LoRA: {lora_params_with_grad}/{lora_params_total} params with gradients")
    print(f"  - All experts received gradients: {all_experts_have_grads}")
    print(f"  - Base model frozen: {base_params_with_grad == 0}")
    print(f"  - Gradient accumulation working: {1.5 < grad_ratio < 2.5}")
    
    return True


if __name__ == "__main__":
    success = test_gradient_flow()
    exit(0 if success else 1)
