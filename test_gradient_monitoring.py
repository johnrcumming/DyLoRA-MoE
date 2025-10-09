#!/usr/bin/env python3
"""
Test script to verify per-expert gradient monitoring works correctly.
"""

import torch
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import Dataset
import sys

from dylo_moe.model import DyLoRA_MoE

# Import the callback from train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import GradientMonitoringCallback


class TestGradientCapture(TrainerCallback):
    """Callback to capture gradient stats for testing."""
    
    def __init__(self):
        self.gradient_logs = []
        
    def on_step_end(self, args, state, control, **kwargs):
        """Capture the last logged gradients."""
        # This will be called after GradientMonitoringCallback
        pass


def test_gradient_monitoring():
    print("="*80)
    print("GRADIENT MONITORING TEST")
    print("="*80)
    
    # 1. Initialize model
    print("\n1. Initializing model with 4 experts...")
    model_name = "google/codegemma-2b"
    hf_token = os.environ.get("HF_TOKEN")
    
    model = DyLoRA_MoE(
        model_name,
        num_experts=4,
        token=hf_token,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        allow_expert_growth=False
    )
    
    # Move model to MPS device if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print(f"Using MPS device")
    else:
        device = model.foundation_model.device
        print(f"MPS not available, using: {device}")
    
    model.train()
    print(f"Model on device: {device}")
    print(f"Number of experts: {model.expert_manager.num_experts}")
    
    # 2. Create small datasets
    print("\n2. Creating dummy datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    texts = ["def hello(): return 'world'" for _ in range(8)]
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    
    train_data = {
        "input_ids": inputs.input_ids[:6],
        "attention_mask": inputs.attention_mask[:6],
    }
    train_data["labels"] = train_data["input_ids"].clone()
    train_dataset = Dataset.from_dict({k: v.tolist() for k, v in train_data.items()})
    
    eval_data = {
        "input_ids": inputs.input_ids[6:],
        "attention_mask": inputs.attention_mask[6:],
    }
    eval_data["labels"] = eval_data["input_ids"].clone()
    eval_dataset = Dataset.from_dict({k: v.tolist() for k, v in eval_data.items()})
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Eval dataset: {len(eval_dataset)} examples")
    
    # 3. Test the callback
    print("\n3. Testing GradientMonitoringCallback...")
    
    # Create a mock gradient logging dict
    gradient_logs = []
    
    class MockWandb:
        @staticmethod
        def log(log_dict, step=None):
            gradient_logs.append(log_dict)
    
    # Temporarily replace wandb
    import train
    original_wandb = train.wandb
    train.wandb = MockWandb()
    
    try:
        # Configure trainer with gradient monitoring
        training_args = TrainingArguments(
            output_dir="./test_grad_monitor",
            max_steps=3,  # Only train for 3 steps
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=1,  # Log every step
            report_to="none",
        )
        
        grad_callback = GradientMonitoringCallback(model, num_experts=4)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[grad_callback],
        )
        
        # 4. Run a few training steps
        print("\n4. Running 3 training steps...")
        trainer.train()
        
        # 5. Check captured gradients
        print("\n5. Analyzing captured gradient logs...")
        print(f"Number of gradient logs captured: {len(gradient_logs)}")
        
        if len(gradient_logs) == 0:
            print("❌ No gradient logs captured!")
            return False
        
        # Check the first log
        first_log = gradient_logs[0]
        print(f"\nFirst gradient log keys: {list(first_log.keys())}")
        
        expected_keys = [
            "grad_norm/total",
            "grad_norm/router",
            "grad_norm/expert_0",
            "grad_norm/expert_1", 
            "grad_norm/expert_2",
            "grad_norm/expert_3",
        ]
        
        issues = []
        for key in expected_keys:
            if key in first_log:
                value = first_log[key]
                print(f"✅ {key}: {value:.6f}")
                if value == 0:
                    issues.append(f"⚠️  {key} is zero")
            else:
                issues.append(f"❌ Missing key: {key}")
                print(f"❌ Missing key: {key}")
        
        # Check gradient ratios
        ratio_keys = [k for k in first_log.keys() if k.startswith("grad_ratio/")]
        if ratio_keys:
            print(f"\nGradient ratios found: {len(ratio_keys)}")
            for key in ratio_keys:
                print(f"  {key}: {first_log[key]:.4f}")
        else:
            issues.append("⚠️  No gradient ratios logged")
        
        # 6. Verify gradients across steps
        print("\n6. Checking gradient evolution across steps...")
        for i, log in enumerate(gradient_logs):
            total_norm = log.get("grad_norm/total", 0)
            print(f"Step {i+1}: Total grad norm = {total_norm:.6f}")
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        if not issues:
            print("🎉 ALL CHECKS PASSED!")
            print("✅ Gradient monitoring callback works correctly")
            print("✅ Per-expert gradients logged")
            print("✅ Router gradients logged")
            print("✅ Gradient ratios computed")
            return True
        else:
            print("⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"  {issue}")
            return False
            
    finally:
        # Restore wandb
        train.wandb = original_wandb
        
        # Cleanup
        import shutil
        if os.path.exists("./test_grad_monitor"):
            shutil.rmtree("./test_grad_monitor")


if __name__ == "__main__":
    success = test_gradient_monitoring()
    exit(0 if success else 1)
