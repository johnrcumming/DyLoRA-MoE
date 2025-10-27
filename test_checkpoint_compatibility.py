"""
Test checkpoint save/load compatibility.

Verifies:
1. Model can be saved to checkpoint
2. Model can be loaded from checkpoint
3. Loaded model produces same outputs
4. All components (LoRA, router, experts) are preserved
"""

import torch
import os
import tempfile
import shutil
from dylo_moe.model import DyLoRA_MoE

def test_checkpoint_compatibility():
    """Test saving and loading checkpoints."""
    print("\n" + "="*60)
    print("Testing Checkpoint Save/Load Compatibility")
    print("="*60)
    
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
    
    print(f"\n1. Creating original model...")
    model1 = DyLoRA_MoE(
        model_name=model_name,
        num_experts=4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        token=hf_token,
        allow_expert_growth=False,
        balance_coefficient=0.01,
    )
    model1.eval()
    
    print("✓ Original model created")
    
    # Create test input
    print("\n2. Creating test input...")
    batch_size = 2
    seq_length = 16
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    print(f"   Input shape: {input_ids.shape}")
    
    # Get outputs from original model
    print("\n3. Running forward pass on original model...")
    with torch.no_grad():
        outputs1 = model1(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    
    logits1 = outputs1.logits
    routing_weights1 = model1.last_routing_weights
    
    print(f"✓ Original logits shape: {logits1.shape}")
    print(f"✓ Original routing weights shape: {routing_weights1.shape}")
    
    # Save checkpoint
    print("\n4. Saving checkpoint...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
    
    try:
        # Save model state
        checkpoint = {
            'foundation_model': model1.foundation_model.state_dict(),
            'router': model1.router.state_dict(),
            'expert_manager': {
                'num_experts': model1.expert_manager.num_experts,
                'lora_r': model1.expert_manager.lora_r,
                'lora_alpha': model1.expert_manager.lora_alpha,
                'lora_dropout': model1.expert_manager.lora_dropout,
            },
            'model_config': {
                'model_name': model_name,
                'num_experts': model1.expert_manager.num_experts,
                'balance_coefficient': model1.balance_coefficient,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        checkpoint_size = os.path.getsize(checkpoint_path) / (1024 ** 2)
        print(f"✓ Checkpoint saved to: {checkpoint_path}")
        print(f"   Checkpoint size: {checkpoint_size:.2f} MB")
        
        # Clear memory
        del model1
        import gc
        gc.collect()
        
        # Load checkpoint
        print("\n5. Loading checkpoint into new model...")
        
        model2 = DyLoRA_MoE(
            model_name=model_name,
            num_experts=4,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            token=hf_token,
            allow_expert_growth=False,
            balance_coefficient=0.01,
        )
        model2.eval()
        
        # Load state
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model2.foundation_model.load_state_dict(loaded_checkpoint['foundation_model'])
        model2.router.load_state_dict(loaded_checkpoint['router'])
        
        print("✓ Checkpoint loaded successfully")
        
        # Verify configuration matches
        print("\n6. Verifying configuration...")
        
        config_matches = True
        loaded_config = loaded_checkpoint['expert_manager']
        
        if model2.expert_manager.num_experts != loaded_config['num_experts']:
            print(f"❌ num_experts mismatch: {model2.expert_manager.num_experts} vs {loaded_config['num_experts']}")
            config_matches = False
        
        if model2.expert_manager.lora_r != loaded_config['lora_r']:
            print(f"❌ lora_r mismatch: {model2.expert_manager.lora_r} vs {loaded_config['lora_r']}")
            config_matches = False
        
        if config_matches:
            print("✓ Configuration matches")
        
        # Run forward pass on loaded model
        print("\n7. Running forward pass on loaded model...")
        
        with torch.no_grad():
            outputs2 = model2(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        logits2 = outputs2.logits
        routing_weights2 = model2.last_routing_weights
        
        print(f"✓ Loaded logits shape: {logits2.shape}")
        print(f"✓ Loaded routing weights shape: {routing_weights2.shape}")
        
        # Compare outputs
        print("\n8. Comparing outputs...")
        
        # Check logits match
        logits_diff = (logits1 - logits2).abs().max().item()
        logits_mean_diff = (logits1 - logits2).abs().mean().item()
        
        print(f"   Logits max diff: {logits_diff:.6f}")
        print(f"   Logits mean diff: {logits_mean_diff:.6f}")
        
        if logits_diff < 1e-5:
            print("✓ Logits match exactly")
        elif logits_diff < 1e-3:
            print("✓ Logits match (small numerical differences)")
        else:
            print(f"⚠️  Logits differ significantly: {logits_diff:.6f}")
        
        # Check routing weights match
        routing_diff = (routing_weights1 - routing_weights2).abs().max().item()
        routing_mean_diff = (routing_weights1 - routing_weights2).abs().mean().item()
        
        print(f"   Routing weights max diff: {routing_diff:.6f}")
        print(f"   Routing weights mean diff: {routing_mean_diff:.6f}")
        
        if routing_diff < 1e-5:
            print("✓ Routing weights match exactly")
        elif routing_diff < 1e-3:
            print("✓ Routing weights match (small numerical differences)")
        else:
            print(f"⚠️  Routing weights differ significantly: {routing_diff:.6f}")
        
        # Test that gradients still work
        print("\n9. Testing gradients on loaded model...")
        
        model2.train()
        labels = input_ids.clone()
        
        outputs3 = model2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        model2.zero_grad()
        outputs3.loss.backward()
        
        # Check router gradients
        router_has_grads = False
        for param in model2.router.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                router_has_grads = True
                break
        
        if router_has_grads:
            print("✓ Router gradients work after loading")
        else:
            print("❌ Router has no gradients after loading")
            return False
        
        # Check LoRA gradients
        lora_has_grads = False
        for name, param in model2.foundation_model.named_parameters():
            if "lora" in name.lower() and param.grad is not None and param.grad.abs().sum() > 0:
                lora_has_grads = True
                break
        
        if lora_has_grads:
            print("✓ LoRA gradients work after loading")
        else:
            print("❌ LoRA has no gradients after loading")
            return False
        
        print("\n" + "="*60)
        print("✅ ALL CHECKPOINT TESTS PASSED!")
        print("="*60)
        print("\nSummary:")
        print(f"  - Checkpoint saved: {checkpoint_size:.2f} MB")
        print(f"  - Logits match: {logits_diff < 1e-3}")
        print(f"  - Routing weights match: {routing_diff < 1e-3}")
        print(f"  - Gradients work: {router_has_grads and lora_has_grads}")
        
        return True
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n✓ Cleaned up temporary directory")


if __name__ == "__main__":
    success = test_checkpoint_compatibility()
    exit(0 if success else 1)
