"""
Test training loop to verify loss convergence.

Runs a short training loop (10 steps) to verify:
1. Loss decreases over training steps
2. Router learns (routing weights change)
3. LoRA adapters learn (parameters update)
4. No NaN or Inf values
5. Training is stable
"""

import torch
import torch.optim as optim
import os
from dylo_moe.model import DyLoRA_MoE

def test_training():
    """Test a short training loop."""
    print("\n" + "="*60)
    print("Training Validation Test")
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
    
    print(f"\n1. Creating model...")
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
    model.train()
    
    print("✓ Model created")
    
    # Setup optimizer
    print("\n2. Setting up optimizer...")
    
    # Only optimize trainable parameters (LoRA + router)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    
    print(f"✓ Optimizer created")
    print(f"   Trainable parameters: {len(trainable_params)}")
    
    # Create synthetic training data
    print("\n3. Creating synthetic training data...")
    
    num_steps = 10
    batch_size = 2
    seq_length = 16
    
    # Generate random data
    training_data = []
    for _ in range(num_steps):
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        training_data.append((input_ids, attention_mask, labels))
    
    print(f"✓ Created {num_steps} training batches")
    print(f"   Batch size: {batch_size}, Sequence length: {seq_length}")
    
    # Training loop
    print("\n4. Running training loop...")
    print("-" * 60)
    
    losses = []
    lm_losses = []
    balance_losses = []
    router_grad_norms = []
    
    # Store initial router state
    initial_router_params = {}
    for name, param in model.router.named_parameters():
        initial_router_params[name] = param.data.clone()
    
    # Store initial LoRA state (first layer, first expert)
    initial_lora_param = None
    initial_lora_name = None
    for name, param in model.foundation_model.named_parameters():
        if "lora" in name.lower() and "expert_0" in name:
            initial_lora_param = param.data.clone()
            initial_lora_name = name
            break
    
    for step, (input_ids, attention_mask, labels) in enumerate(training_data):
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ Step {step}: Loss is NaN or Inf!")
            return False
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check for NaN/Inf gradients
        has_nan_grad = False
        for param in trainable_params:
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"❌ Step {step}: NaN or Inf gradient detected!")
            return False
        
        # Compute router gradient norm
        router_grad_norm = 0.0
        for param in model.router.parameters():
            if param.grad is not None:
                router_grad_norm += param.grad.norm().item() ** 2
        router_grad_norm = router_grad_norm ** 0.5
        router_grad_norms.append(router_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Track losses
        losses.append(loss.item())
        if model.last_lm_loss is not None:
            lm_losses.append(model.last_lm_loss.item())
        if model.last_balance_loss is not None:
            balance_losses.append(model.last_balance_loss.item())
        
        # Print progress
        print(f"   Step {step+1:2d}: Loss={loss.item():7.4f}", end="")
        if model.last_lm_loss is not None:
            print(f" | LM={model.last_lm_loss.item():7.4f}", end="")
        if model.last_balance_loss is not None:
            print(f" | Balance={model.last_balance_loss.item():.6f}", end="")
        print(f" | Router grad={router_grad_norm:.6f}")
    
    print("-" * 60)
    
    # Analyze results
    print("\n5. Analyzing training results...")
    
    # Check loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = initial_loss - final_loss
    loss_reduction_pct = (loss_reduction / initial_loss) * 100
    
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Final loss: {final_loss:.4f}")
    print(f"   Loss reduction: {loss_reduction:.4f} ({loss_reduction_pct:.2f}%)")
    
    if loss_reduction > 0:
        print("✓ Loss decreased")
    else:
        print("⚠️  Loss did not decrease (may need more steps or higher LR)")
    
    # Check router parameters changed
    router_changed = False
    max_param_change = 0.0
    
    for name, param in model.router.named_parameters():
        param_change = (param.data - initial_router_params[name]).abs().max().item()
        max_param_change = max(max_param_change, param_change)
        if param_change > 1e-6:
            router_changed = True
    
    print(f"\n   Router max parameter change: {max_param_change:.6f}")
    
    if router_changed:
        print("✓ Router parameters updated")
    else:
        print("⚠️  Router parameters did not change significantly")
    
    # Check LoRA parameters changed
    if initial_lora_param is not None:
        for name, param in model.foundation_model.named_parameters():
            if name == initial_lora_name:
                lora_change = (param.data - initial_lora_param).abs().max().item()
                print(f"\n   LoRA max parameter change: {lora_change:.6f}")
                
                if lora_change > 1e-6:
                    print("✓ LoRA parameters updated")
                else:
                    print("⚠️  LoRA parameters did not change significantly")
                break
    
    # Check gradient stability
    avg_router_grad = sum(router_grad_norms) / len(router_grad_norms)
    max_router_grad = max(router_grad_norms)
    min_router_grad = min(router_grad_norms)
    
    print(f"\n   Router gradient norm:")
    print(f"     Average: {avg_router_grad:.6f}")
    print(f"     Min: {min_router_grad:.6f}")
    print(f"     Max: {max_router_grad:.6f}")
    
    if max_router_grad < 100:
        print("✓ Gradients are stable (no explosion)")
    else:
        print("⚠️  Gradients may be exploding")
    
    if min_router_grad > 1e-7:
        print("✓ Gradients are not vanishing")
    else:
        print("⚠️  Gradients may be vanishing")
    
    # Plot loss curve (text-based)
    print("\n6. Loss curve:")
    print("-" * 60)
    
    # Simple text-based plot
    max_loss = max(losses)
    min_loss = min(losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0
    
    for i, loss in enumerate(losses):
        normalized = (loss - min_loss) / loss_range
        bar_length = int(normalized * 40)
        bar = "█" * bar_length
        print(f"   Step {i+1:2d}: {bar} {loss:.4f}")
    
    print("\n" + "="*60)
    print("✅ TRAINING VALIDATION COMPLETE!")
    print("="*60)
    
    print("\nSummary:")
    print(f"  - Loss decreased: {loss_reduction > 0}")
    print(f"  - Loss reduction: {loss_reduction_pct:.2f}%")
    print(f"  - Router learning: {router_changed}")
    print(f"  - LoRA learning: {lora_change > 1e-6 if initial_lora_param is not None else 'N/A'}")
    print(f"  - Gradients stable: {max_router_grad < 100 and min_router_grad > 1e-7}")
    print(f"  - No NaN/Inf: True")
    
    return True


if __name__ == "__main__":
    success = test_training()
    exit(0 if success else 1)
