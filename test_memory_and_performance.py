"""
Benchmark memory usage and performance of the new single-pass routing.

Tests:
1. Peak memory usage during forward pass
2. Peak memory usage during forward + backward pass
3. Forward pass time
4. Backward pass time
"""

import torch
import os
import time
import gc
from dylo_moe.model import DyLoRA_MoE

def get_memory_allocated():
    """Get currently allocated memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        # CPU memory is harder to track precisely
        return 0

def get_peak_memory():
    """Get peak memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        return 0

def reset_peak_memory():
    """Reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def run_benchmark():
    """Run memory and performance benchmarks."""
    print("\n" + "="*60)
    print("Memory & Performance Benchmark")
    print("="*60)
    
    # Use tiny model
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
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = DyLoRA_MoE(
        model_name=model_name,
        num_experts=4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,  # Disable dropout for consistent timing
        token=hf_token,
        allow_expert_growth=False,
        balance_coefficient=0.01,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    if device == "cuda":
        model = model.cuda()
        print(f"   Model on GPU")
    
    model.train()
    
    # Model memory
    model_memory = get_memory_allocated()
    print(f"\n2. Model memory: {model_memory:.2f} MB")
    
    # Create test inputs with varying batch sizes
    batch_sizes = [1, 2, 4, 8]
    seq_length = 32
    
    print(f"\n3. Benchmarking with sequence length: {seq_length}")
    print("-" * 60)
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n   Batch size: {batch_size}")
        
        # Create inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        if device == "cuda":
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
        
        # Warmup run
        _ = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model.zero_grad()
        
        # Reset peak memory tracking
        reset_peak_memory()
        
        # Forward pass timing
        start_time = time.time()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Peak memory after forward
        forward_memory = get_peak_memory()
        
        # Backward pass timing
        reset_peak_memory()
        start_time = time.time()
        outputs.loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        backward_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Peak memory after backward
        backward_memory = get_peak_memory()
        
        results.append({
            'batch_size': batch_size,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'forward_memory': forward_memory,
            'backward_memory': backward_memory,
            'total_time': forward_time + backward_time
        })
        
        print(f"     Forward time:  {forward_time:6.2f} ms")
        print(f"     Backward time: {backward_time:6.2f} ms")
        print(f"     Total time:    {forward_time + backward_time:6.2f} ms")
        
        if device == "cuda":
            print(f"     Forward memory:  {forward_memory:.2f} MB")
            print(f"     Backward memory: {backward_memory:.2f} MB")
        
        # Cleanup
        del outputs
        model.zero_grad()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary table
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    print(f"\n{'Batch':>6} | {'Forward (ms)':>12} | {'Backward (ms)':>13} | {'Total (ms)':>10}", end="")
    if device == "cuda":
        print(f" | {'Memory (MB)':>11}")
    else:
        print()
    print("-" * 60)
    
    for r in results:
        print(f"{r['batch_size']:>6} | {r['forward_time']:>12.2f} | {r['backward_time']:>13.2f} | {r['total_time']:>10.2f}", end="")
        if device == "cuda":
            print(f" | {r['backward_memory']:>11.2f}")
        else:
            print()
    
    # Calculate throughput (sequences per second)
    print("\n" + "="*60)
    print("Throughput (sequences/second)")
    print("="*60)
    
    for r in results:
        throughput = (r['batch_size'] / r['total_time']) * 1000  # Convert ms to s
        print(f"   Batch {r['batch_size']:>2}: {throughput:>6.2f} seq/s")
    
    # Scaling analysis
    print("\n" + "="*60)
    print("Scaling Analysis")
    print("="*60)
    
    # Compare batch 1 vs batch 8
    if len(results) >= 2:
        small_batch = results[0]
        large_batch = results[-1]
        
        time_ratio = large_batch['total_time'] / small_batch['total_time']
        batch_ratio = large_batch['batch_size'] / small_batch['batch_size']
        efficiency = batch_ratio / time_ratio
        
        print(f"   Batch size increase: {small_batch['batch_size']}x → {large_batch['batch_size']}x ({batch_ratio:.1f}x)")
        print(f"   Time increase: {small_batch['total_time']:.2f}ms → {large_batch['total_time']:.2f}ms ({time_ratio:.2f}x)")
        print(f"   Scaling efficiency: {efficiency:.2f} (1.0 = perfect linear scaling)")
        
        if efficiency > 0.8:
            print("   ✓ Good scaling efficiency")
        elif efficiency > 0.6:
            print("   ⚠️  Moderate scaling efficiency")
        else:
            print("   ⚠️  Poor scaling efficiency")
    
    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETE")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = run_benchmark()
    exit(0 if success else 1)
