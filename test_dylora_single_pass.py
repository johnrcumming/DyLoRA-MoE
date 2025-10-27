"""
Test DyLoRA_MoE single-pass routing implementation.

This test validates that the refactored forward pass using PEFT's routing_weights
parameter works correctly and produces valid outputs.
"""

import torch
import os
from dylo_moe.model import DyLoRA_MoE

def test_single_pass_forward():
    """Test that forward pass completes without errors with multi-expert routing."""
    print("\n" + "="*60)
    print("Testing DyLoRA_MoE Single-Pass Forward")
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
    
    print("✓ Model initialized successfully")
    print(f"   - Number of experts: {model.expert_manager.num_experts}")
    print(f"   - Router num_experts: {model.router.num_experts}")
    
    # Create test input
    print("\n2. Creating test input...")
    batch_size = 2
    seq_length = 16
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    print(f"   - Input shape: {input_ids.shape}")
    print(f"   - Batch size: {batch_size}, Sequence length: {seq_length}")
    
    # Test forward pass (multi-expert routing)
    print("\n3. Running forward pass with multi-expert routing...")
    
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        print("✓ Forward pass completed successfully")
    except Exception as e:
        print(f"❌ Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate outputs
    print("\n4. Validating outputs...")
    
    # Check logits shape
    expected_logits_shape = (batch_size, seq_length, model.foundation_model.config.vocab_size)
    if outputs.logits.shape != expected_logits_shape:
        print(f"❌ Logits shape mismatch!")
        print(f"   Expected: {expected_logits_shape}")
        print(f"   Got: {outputs.logits.shape}")
        return False
    print(f"✓ Logits shape correct: {outputs.logits.shape}")
    
    # Check loss exists
    if outputs.loss is None:
        print("❌ Loss is None!")
        return False
    print(f"✓ Loss computed: {outputs.loss.item():.4f}")
    
    # Check routing weights were tracked
    if model.last_routing_weights is None:
        print("❌ Routing weights not tracked!")
        return False
    
    expected_routing_shape = (batch_size, seq_length, model.router.num_experts)
    if model.last_routing_weights.shape != expected_routing_shape:
        print(f"❌ Routing weights shape mismatch!")
        print(f"   Expected: {expected_routing_shape}")
        print(f"   Got: {model.last_routing_weights.shape}")
        return False
    print(f"✓ Routing weights tracked: {model.last_routing_weights.shape}")
    
    # Check routing weights are normalized (sum to ~1)
    routing_sums = model.last_routing_weights.sum(dim=-1)  # Sum over experts
    if not torch.allclose(routing_sums, torch.ones_like(routing_sums), atol=1e-5):
        print(f"❌ Routing weights not normalized!")
        print(f"   Sum range: [{routing_sums.min():.6f}, {routing_sums.max():.6f}]")
        return False
    print(f"✓ Routing weights normalized (sum to 1)")
    
    # Check load balancing loss was computed
    if model.last_balance_loss is None:
        print("⚠️  Load balancing loss not computed (expected during training)")
    else:
        print(f"✓ Load balancing loss: {model.last_balance_loss.item():.6f}")
    
    print("\n5. Testing backward pass...")
    try:
        outputs.loss.backward()
        print("✓ Backward pass completed successfully")
    except Exception as e:
        print(f"❌ Backward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check gradients exist for router
    router_has_grads = False
    for param in model.router.parameters():
        if param.grad is not None:
            router_has_grads = True
            break
    
    if not router_has_grads:
        print("❌ Router has no gradients!")
        return False
    print("✓ Router gradients computed")
    
    # Check gradients exist for LoRA adapters
    lora_has_grads = False
    for name, param in model.foundation_model.named_parameters():
        if "lora" in name.lower() and param.grad is not None:
            lora_has_grads = True
            break
    
    if not lora_has_grads:
        print("❌ LoRA adapters have no gradients!")
        return False
    print("✓ LoRA adapter gradients computed")
    
    print("\n6. Testing single expert mode...")
    model.zero_grad()
    
    try:
        outputs_single = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            expert_id=0,  # Train only expert 0
        )
        print("✓ Single expert mode works")
        print(f"   Loss: {outputs_single.loss.item():.4f}")
    except Exception as e:
        print(f"❌ Single expert mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_single_pass_forward()
    exit(0 if success else 1)
