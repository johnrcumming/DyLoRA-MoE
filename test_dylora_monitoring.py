"""
Test script to validate DyLoRA monitoring callback functionality.
Tests routing statistics tracking, expert usage patterns, and entropy calculations.
"""
import torch
import math
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import sys

# Mock wandb for testing - create a module-like object
class MockWandbModule:
    """Mock wandb module that captures all log calls."""
    def __init__(self):
        self.logs = []
        self.is_initialized = False
    
    def log(self, log_dict, step=None):
        """Capture wandb.log() calls."""
        self.logs.append({
            'step': step,
            'metrics': log_dict.copy()
        })
    
    def init(self, **kwargs):
        """Mock wandb.init()."""
        self.is_initialized = True
        return self
    
    def finish(self):
        """Mock wandb.finish()."""
        pass

# Create and install mock BEFORE importing train module
mock_wandb = MockWandbModule()
sys.modules['wandb'] = mock_wandb

# Now import after wandb is mocked
from dylo_moe.model import DyLoRA_MoE
from train import DyLoRAMonitoringCallback


def create_test_dataset(tokenizer, num_samples=6):
    """Create a small test dataset."""
    texts = [
        "def hello_world():\n    print('Hello, World!')",
        "Write a function to calculate factorial",
        "class MyClass:\n    def __init__(self):\n        pass",
        "for i in range(10):\n    print(i)",
        "import numpy as np\nx = np.array([1, 2, 3])",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
    ]
    
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"],
    })
    
    return dataset


def test_dylora_monitoring():
    """Test the DyLoRA monitoring callback."""
    print("=" * 80)
    print("Testing DyLoRA Monitoring Callback")
    print("=" * 80)
    
    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Initialize model
    model_name = "google/gemma-3-270m"
    num_experts = 4
    
    print(f"\nInitializing DyLoRA-MoE model...")
    print(f"Base model: {model_name}")
    print(f"Number of experts: {num_experts}")
    
    model = DyLoRA_MoE(
        model_name=model_name,
        num_experts=num_experts,
        lora_r=8,  # Smaller for testing
        lora_alpha=16,
        lora_dropout=0.05,
        allow_expert_growth=False
    )
    
    # Mark all experts as mature
    for i in range(num_experts):
        model.router.set_expert_maturity(i, 1)
    
    model = model.to(device)
    print(f"Model on device: {model.foundation_model.device}")
    
    # Create tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = create_test_dataset(tokenizer, num_samples=6)
    print(f"Train dataset: {len(train_dataset)} examples")
    
    # Create callback
    dylora_callback = DyLoRAMonitoringCallback(
        model=model,
        num_experts=num_experts
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./test_dylora_monitoring_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,  # Log every step for testing
        logging_strategy="steps",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        max_steps=3,  # Only run 3 steps for testing
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[dylora_callback],
    )
    
    # Run training
    print("\nRunning 3 training steps...")
    trainer.train()
    
    # Analyze results
    print("\n" + "=" * 80)
    print("Analyzing DyLoRA Monitoring Results")
    print("=" * 80)
    
    routing_logs = [log for log in mock_wandb.logs if any(k.startswith('routing/') for k in log['metrics'].keys())]
    
    print(f"\nNumber of routing logs captured: {len(routing_logs)}")
    
    if len(routing_logs) == 0:
        print("❌ ERROR: No routing statistics were logged!")
        print("\nDebugging information:")
        print(f"Total logs: {len(mock_wandb.logs)}")
        print(f"Has last_routing_weights: {hasattr(model, 'last_routing_weights')}")
        if hasattr(model, 'last_routing_weights'):
            print(f"Last routing weights is None: {model.last_routing_weights is None}")
        return False
    
    # Check the latest routing log
    latest_log = routing_logs[-1]
    metrics = latest_log['metrics']
    
    print("\n" + "=" * 80)
    print("Routing Statistics (Latest Step)")
    print("=" * 80)
    
    # 1. Expert Usage Distribution
    print("\n1. Expert Usage Distribution:")
    for i in range(num_experts):
        usage_key = f'routing/expert_{i}_usage'
        if usage_key in metrics:
            print(f"   Expert {i}: {metrics[usage_key]:.4f} ({metrics[usage_key]*100:.2f}%)")
    
    # 2. Entropy Metrics
    print("\n2. Routing Entropy:")
    if 'routing/entropy' in metrics:
        print(f"   Raw Entropy: {metrics['routing/entropy']:.4f}")
        print(f"   Normalized Entropy: {metrics['routing/entropy_normalized']:.4f}")
        print(f"   Max Possible Entropy: {math.log(num_experts):.4f}")
        
        # Interpretation
        normalized_entropy = metrics['routing/entropy_normalized']
        if normalized_entropy > 0.9:
            print("   → Very balanced routing (experts used evenly)")
        elif normalized_entropy > 0.7:
            print("   → Moderately balanced routing")
        elif normalized_entropy > 0.5:
            print("   → Some expert preference emerging")
        else:
            print("   → Strong expert specialization/concentration")
    
    # 3. Load Balancing
    print("\n3. Load Balancing:")
    if 'routing/load_imbalance' in metrics:
        print(f"   Load Imbalance (CV): {metrics['routing/load_imbalance']:.4f}")
        if metrics['routing/load_imbalance'] < 0.3:
            print("   → Well balanced")
        elif metrics['routing/load_imbalance'] < 0.6:
            print("   → Moderately imbalanced")
        else:
            print("   → Highly imbalanced")
    
    # 4. Expert Dominance
    print("\n4. Expert Dominance:")
    if 'routing/max_expert_usage' in metrics:
        max_usage = metrics['routing/max_expert_usage']
        print(f"   Max Expert Usage: {max_usage:.4f} ({max_usage*100:.2f}%)")
        if max_usage > 0.8:
            print("   → Strong dominance by one expert")
        elif max_usage > 0.5:
            print("   → Moderate dominance")
        else:
            print("   → No single expert dominates")
    
    # 5. Active Experts
    print("\n5. Active Experts:")
    if 'routing/active_experts' in metrics:
        active = int(metrics['routing/active_experts'])
        print(f"   Active Experts (>5% usage): {active}/{num_experts}")
        print(f"   Utilization: {active/num_experts*100:.1f}%")
    
    # 6. Routing Confidence
    print("\n6. Routing Confidence:")
    if 'routing/avg_max_confidence' in metrics:
        confidence = metrics['routing/avg_max_confidence']
        print(f"   Avg Max Confidence: {confidence:.4f}")
        if confidence > 0.8:
            print("   → High confidence (router is decisive)")
        elif confidence > 0.5:
            print("   → Moderate confidence")
        else:
            print("   → Low confidence (distributed routing)")
    
    # 7. Cumulative Usage
    print("\n7. Cumulative Expert Usage:")
    cumulative_found = False
    for i in range(num_experts):
        cumulative_key = f'routing/cumulative_expert_{i}'
        if cumulative_key in metrics:
            cumulative_found = True
            print(f"   Expert {i}: {metrics[cumulative_key]:.4f} ({metrics[cumulative_key]*100:.2f}%)")
    
    if not cumulative_found:
        print("   (Not yet available - needs multiple steps)")
    
    # 8. Token Variance
    print("\n8. Expert Token Variance:")
    for i in range(num_experts):
        variance_key = f'routing/expert_{i}_token_variance'
        if variance_key in metrics:
            print(f"   Expert {i}: {metrics[variance_key]:.6f}")
    
    # Validation checks
    print("\n" + "=" * 80)
    print("Validation Checks")
    print("=" * 80)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Expert usage sums to ~1.0
    total_checks += 1
    expert_usage_sum = sum(metrics.get(f'routing/expert_{i}_usage', 0) for i in range(num_experts))
    if abs(expert_usage_sum - 1.0) < 0.01:
        print(f"✅ Expert usage sums to 1.0: {expert_usage_sum:.4f}")
        checks_passed += 1
    else:
        print(f"❌ Expert usage doesn't sum to 1.0: {expert_usage_sum:.4f}")
    
    # Check 2: Entropy is within valid range
    total_checks += 1
    if 'routing/entropy' in metrics:
        entropy = metrics['routing/entropy']
        max_entropy = math.log(num_experts)
        if 0 <= entropy <= max_entropy:
            print(f"✅ Entropy in valid range: {entropy:.4f} <= {max_entropy:.4f}")
            checks_passed += 1
        else:
            print(f"❌ Entropy out of range: {entropy:.4f} (max: {max_entropy:.4f})")
    
    # Check 3: Normalized entropy in [0, 1]
    total_checks += 1
    if 'routing/entropy_normalized' in metrics:
        norm_entropy = metrics['routing/entropy_normalized']
        if 0 <= norm_entropy <= 1:
            print(f"✅ Normalized entropy in [0,1]: {norm_entropy:.4f}")
            checks_passed += 1
        else:
            print(f"❌ Normalized entropy out of [0,1]: {norm_entropy:.4f}")
    
    # Check 4: At least one expert is active
    total_checks += 1
    if 'routing/active_experts' in metrics:
        active = metrics['routing/active_experts']
        if active >= 1:
            print(f"✅ At least one expert is active: {int(active)}")
            checks_passed += 1
        else:
            print(f"❌ No experts are active: {int(active)}")
    
    # Check 5: Max confidence is reasonable
    total_checks += 1
    if 'routing/avg_max_confidence' in metrics:
        confidence = metrics['routing/avg_max_confidence']
        if 0 <= confidence <= 1:
            print(f"✅ Max confidence in valid range: {confidence:.4f}")
            checks_passed += 1
        else:
            print(f"❌ Max confidence out of range: {confidence:.4f}")
    
    # Check 6: All expected metrics are present
    total_checks += 1
    expected_metrics = [
        'routing/entropy',
        'routing/entropy_normalized',
        'routing/load_imbalance',
        'routing/max_expert_usage',
        'routing/active_experts',
        'routing/avg_max_confidence',
    ]
    missing_metrics = [m for m in expected_metrics if m not in metrics]
    if not missing_metrics:
        print(f"✅ All expected metrics present")
        checks_passed += 1
    else:
        print(f"❌ Missing metrics: {missing_metrics}")
    
    print("\n" + "=" * 80)
    print(f"Validation Summary: {checks_passed}/{total_checks} checks passed")
    print("=" * 80)
    
    if checks_passed == total_checks:
        print("\n✅ DyLoRA monitoring callback is working correctly!")
        return True
    else:
        print(f"\n⚠️  {total_checks - checks_passed} checks failed. Review the output above.")
        return False


if __name__ == "__main__":
    success = test_dylora_monitoring()
    exit(0 if success else 1)
