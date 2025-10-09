#!/usr/bin/env python3
"""
Test script to verify evaluation metric naming is fixed.
This script checks that the trainer produces the correct metric names for early stopping.
"""

import torch
import os
os.environ["WANDB_DISABLED"] = "true"
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset

from dylo_moe.model import DyLoRA_MoE

def test_eval_metrics():
    print("="*80)
    print("EVALUATION METRIC NAMING TEST")
    print("="*80)
    
    # 1. Initialize small model
    print("\n1. Initializing model...")
    model_name = "google/codegemma-2b"
    hf_token = os.environ.get("HF_TOKEN")
    
    model = DyLoRA_MoE(
        model_name,
        num_experts=2,  # Small for testing
        token=hf_token,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        allow_expert_growth=False
    )
    
    model.train()
    device = model.foundation_model.device
    print(f"Model loaded on device: {device}")
    
    # 2. Create small datasets
    print("\n2. Creating dummy datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create small dummy datasets
    texts = [
        "def hello(): return 'world'",
        "class Test: pass",
        "import numpy as np",
        "x = [1, 2, 3]"
    ]
    
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    
    # Create datasets
    train_data = {
        "input_ids": inputs.input_ids[:3],
        "attention_mask": inputs.attention_mask[:3],
    }
    train_data["labels"] = train_data["input_ids"].clone()
    train_dataset = Dataset.from_dict({k: v.tolist() for k, v in train_data.items()})
    
    eval_data = {
        "input_ids": inputs.input_ids[3:],
        "attention_mask": inputs.attention_mask[3:],
    }
    eval_data["labels"] = eval_data["input_ids"].clone()
    eval_dataset = Dataset.from_dict({k: v.tolist() for k, v in eval_data.items()})
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Eval dataset: {len(eval_dataset)} examples")
    
    # 3. Configure trainer with early stopping
    print("\n3. Configuring trainer...")
    training_args = TrainingArguments(
        output_dir="./test_results",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="no",
        logging_steps=1,
        metric_for_best_model="loss",  # Fixed: use "loss" not "eval_loss"
        greater_is_better=False,
        load_best_model_at_end=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    # 4. Test evaluation
    print("\n4. Running evaluation...")
    eval_results = trainer.evaluate()
    
    print("\n5. Checking metric names...")
    print(f"Metrics returned: {list(eval_results.keys())}")
    
    # Check for expected metrics
    expected_metrics = ["eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"]
    issues = []
    
    for metric in expected_metrics:
        if metric in eval_results:
            print(f"✅ Found '{metric}': {eval_results[metric]}")
        else:
            issues.append(f"❌ Missing '{metric}'")
            print(f"❌ Missing '{metric}'")
    
    # Check that 'loss' key also exists (for metric_for_best_model)
    # Note: During evaluation, the metric key without prefix might also be present
    if "loss" in eval_results:
        print(f"✅ Found 'loss' (without prefix): {eval_results['loss']}")
    else:
        print(f"ℹ️  'loss' (without prefix) not found - this is OK, 'eval_loss' will be used")
    
    # 6. Test that early stopping callback can find the metric
    print("\n6. Testing early stopping callback...")
    # Find the EarlyStoppingCallback
    early_stopping_callback = None
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, EarlyStoppingCallback):
            early_stopping_callback = callback
            break
    
    if early_stopping_callback:
        print(f"✅ Found EarlyStoppingCallback with patience: {early_stopping_callback.early_stopping_patience}")
    else:
        issues.append("❌ EarlyStoppingCallback not found")
        print("❌ EarlyStoppingCallback not found in trainer")
    
    # Check if trainer state has the metric
    print(f"Trainer's metric_for_best_model: {training_args.metric_for_best_model}")
    
    # The full metric name with prefix should be: eval_{metric_for_best_model}
    full_metric_name = f"eval_{training_args.metric_for_best_model}"
    if full_metric_name in eval_results:
        print(f"✅ Early stopping will work: '{full_metric_name}' found in results")
    else:
        issues.append(f"❌ Early stopping broken: '{full_metric_name}' not in results")
        print(f"❌ Early stopping will fail: '{full_metric_name}' not found")
    
    # 7. Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if not issues:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Evaluation returns correct metric names")
        print("✅ Early stopping will work correctly")
        print("✅ metric_for_best_model configuration is valid")
        return True
    else:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False

if __name__ == "__main__":
    success = test_eval_metrics()
    
    # Cleanup
    import shutil
    if os.path.exists("./test_results"):
        shutil.rmtree("./test_results")
    
    exit(0 if success else 1)
