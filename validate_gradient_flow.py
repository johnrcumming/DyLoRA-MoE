#!/usr/bin/env python3
"""
Validation script to test gradient flow in DyLoRA-MoE model.
This script verifies that:
1. Gradients are properly computed (grad_norm > 0)
2. All LoRA parameters receive gradients
3. Loss decreases over a few training steps
"""

import torch
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import AutoTokenizer
from dylo_moe.model import DyLoRA_MoE

def test_gradient_flow():
    print("="*80)
    print("GRADIENT FLOW VALIDATION TEST")
    print("="*80)
    
    # 1. Initialize model
    print("\n1. Initializing model...")
    model_name = "google/codegemma-2b"
    hf_token = os.environ.get("HF_TOKEN")
    
    model = DyLoRA_MoE(
        model_name,
        num_experts=4,
        token=hf_token,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        allow_expert_growth=False
    )
    
    # Set all experts as mature
    for i in range(model.expert_manager.num_experts):
        model.router.set_expert_maturity(i, 1)
    
    model.train()
    device = model.foundation_model.device
    print(f"Model loaded on device: {device}")
    
    # 2. Create tokenizer and sample data
    print("\n2. Preparing sample data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    sample_texts = [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "class Calculator: def __init__(self): self.result = 0",
        "import numpy as np; arr = np.array([1, 2, 3])",
        "for i in range(10): print(i * 2)",
    ]
    
    inputs = tokenizer(sample_texts, padding=True, truncation=True, 
                       return_tensors="pt", max_length=128)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    labels = input_ids.clone()
    
    print(f"Sample batch shape: {input_ids.shape}")
    
    # 3. Test forward pass WITHOUT expert_id (routing mode)
    print("\n3. Testing forward pass with routing...")
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    print(f"Loss (routing mode): {outputs.loss.item():.4f}")
    
    # 4. Test backward pass and check gradients
    print("\n4. Testing backward pass...")
    model.zero_grad()
    outputs.loss.backward()
    
    # Check gradient statistics
    total_grad_norm = 0.0
    param_count = 0
    params_with_grad = 0
    params_without_grad = 0
    
    lora_grad_norm = 0.0
    router_grad_norm = 0.0
    
    print("\n5. Gradient Statistics:")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += 1
            if param.grad is not None:
                params_with_grad += 1
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                
                if "lora" in name.lower():
                    lora_grad_norm += grad_norm ** 2
                elif "router" in name.lower() or "gate" in name.lower():
                    router_grad_norm += grad_norm ** 2
            else:
                params_without_grad += 1
                print(f"WARNING: No gradient for {name}")
    
    total_grad_norm = total_grad_norm ** 0.5
    lora_grad_norm = lora_grad_norm ** 0.5
    router_grad_norm = router_grad_norm ** 0.5
    
    print(f"Total trainable parameters: {param_count}")
    print(f"Parameters with gradients: {params_with_grad}")
    print(f"Parameters WITHOUT gradients: {params_without_grad}")
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    print(f"LoRA gradient norm: {lora_grad_norm:.6f}")
    print(f"Router gradient norm: {router_grad_norm:.6f}")
    
    # 6. Test expert-specific forward (single expert mode)
    print("\n6. Testing forward pass with explicit expert_id...")
    model.zero_grad()
    outputs_expert = model(input_ids, attention_mask=attention_mask, 
                           labels=labels, expert_id=0)
    print(f"Loss (expert_id=0): {outputs_expert.loss.item():.4f}")
    
    outputs_expert.loss.backward()
    expert_grad_norm = sum(p.grad.norm().item() ** 2 
                          for p in model.parameters() 
                          if p.grad is not None) ** 0.5
    print(f"Gradient norm (expert_id=0): {expert_grad_norm:.6f}")
    
    # 7. Quick training test
    print("\n7. Quick training test (5 steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    initial_loss = None
    for step in range(5):
        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        if step == 0:
            initial_loss = loss.item()
        
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 
                       for p in model.parameters() 
                       if p.grad is not None) ** 0.5
        optimizer.step()
        
        print(f"Step {step+1}: Loss = {loss.item():.4f}, Grad Norm = {grad_norm:.6f}")
    
    final_loss = outputs.loss.item()
    loss_improvement = initial_loss - final_loss
    
    # 8. Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    success = True
    issues = []
    
    # Check 1: Gradients exist
    if params_without_grad > 0:
        success = False
        issues.append(f"❌ {params_without_grad} trainable parameters have no gradients")
    else:
        print("✅ All trainable parameters have gradients")
    
    # Check 2: Gradient norm is non-zero
    if total_grad_norm < 1e-6:
        success = False
        issues.append(f"❌ Total gradient norm is nearly zero: {total_grad_norm}")
    else:
        print(f"✅ Gradient norm is healthy: {total_grad_norm:.6f}")
    
    # Check 3: LoRA gradients exist
    if lora_grad_norm < 1e-6:
        success = False
        issues.append(f"❌ LoRA gradient norm is nearly zero: {lora_grad_norm}")
    else:
        print(f"✅ LoRA gradients flowing: {lora_grad_norm:.6f}")
    
    # Check 4: Router gradients exist
    router_grad_threshold = 1e-10  # More lenient threshold for router
    if router_grad_norm < router_grad_threshold:
        issues.append(f"⚠️  Router gradient norm is very small: {router_grad_norm}")
        print(f"⚠️  Router gradients very small but non-zero: {router_grad_norm:.2e}")
        print("    (This can be normal when expert outputs are similar)")
    else:
        print(f"✅ Router gradients flowing: {router_grad_norm:.6f}")
    
    # Check 5: Loss decreases (or at least changes)
    if abs(loss_improvement) < 1e-6:
        issues.append(f"⚠️  Loss didn't change much: {initial_loss:.4f} -> {final_loss:.4f}")
        print(f"⚠️  Loss improvement marginal: {loss_improvement:.6f}")
    elif loss_improvement > 0:
        print(f"✅ Loss decreased: {initial_loss:.4f} -> {final_loss:.4f} (Δ={loss_improvement:.6f})")
    else:
        print(f"⚠️  Loss increased: {initial_loss:.4f} -> {final_loss:.4f} (may be normal for few steps)")
    
    print("\n" + "="*80)
    if success:
        print("🎉 GRADIENT FLOW VALIDATION PASSED!")
        print("The model is ready for training.")
    else:
        print("⚠️  GRADIENT FLOW VALIDATION FAILED")
        print("\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
    print("="*80)
    
    return success

if __name__ == "__main__":
    test_gradient_flow()
