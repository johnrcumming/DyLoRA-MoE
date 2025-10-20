"""
Base benchmark class for evaluating models on various coding tasks.
"""
import torch
import wandb
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer
from dylo_moe.model import DyLoRA_MoE


class BaseBenchmark(ABC):
    """Abstract base class for model benchmarks."""
    
    def __init__(self, name: str, tokenizer: AutoTokenizer, max_new_tokens: int = 256):
        self.name = name
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        
    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the benchmark dataset."""
        pass
    
    @abstractmethod
    def evaluate_sample(self, model, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample."""
        pass
    
    @abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute final benchmark metrics from all results."""
        pass
    
    def generate_completion(self, model, prompt: str, **generation_kwargs) -> str:
        """Generate code completion for a given prompt."""
        # Default generation parameters
        default_kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'temperature': 0.2,
            'do_sample': True,
            'top_p': 0.95,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        default_kwargs.update(generation_kwargs)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to model's device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            if isinstance(model, DyLoRA_MoE):
                # Use DyLoRA-MoE's generate method (with expert routing)
                outputs = model.generate(**inputs, **default_kwargs)
            else:
                # Standard model generation
                outputs = model.generate(**inputs, **default_kwargs)
        
        # Decode and extract completion
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()
        
        model.train()
        return completion
    
    def run_benchmark(self, model, max_samples: Optional[int] = None, 
                     log_to_wandb: bool = False, prefix: str = "") -> Dict[str, Any]:
        """Run the complete benchmark on a model."""
        print(f"\n{'='*80}")
        print(f"Running {self.name} Benchmark")
        if prefix:
            print(f"Model: {prefix}")
        print(f"{'='*80}")
        
        # Load dataset
        dataset = self.load_dataset()
        if max_samples:
            dataset = dataset[:max_samples]
        
        print(f"Evaluating on {len(dataset)} {self.name} problems...")
        
        # Evaluate each sample
        results = []
        for i, sample in enumerate(dataset):
            try:
                result = self.evaluate_sample(model, sample)
                result['sample_index'] = i
                results.append(result)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{len(dataset)} samples...")
                    
            except Exception as e:
                print(f"  ⚠️  Error on sample {i}: {e}")
                # Add failed result
                results.append({
                    'sample_index': i,
                    'task_id': sample.get('task_id', f'{self.name}_{i}'),
                    'success': False,
                    'error': str(e)
                })
        
        # Compute final metrics
        metrics = self.compute_metrics(results)
        
        # Add metadata
        metrics.update({
            'benchmark_name': self.name,
            'num_samples': len(dataset),
            'num_completed': len([r for r in results if r.get('success', True)]),
            'num_failed': len([r for r in results if not r.get('success', True)]),
        })
        
        # Log to wandb if requested
        if log_to_wandb:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    wandb_key = f"{prefix}/{self.name.lower()}/{key}" if prefix else f"{self.name.lower()}/{key}"
                    wandb_metrics[wandb_key] = value
            
            if wandb_metrics:
                wandb.log(wandb_metrics, commit=False)
        
        # Print results
        print(f"\n{self.name} Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, int):
                print(f"  {key}: {value}")
        
        print(f"{'='*80}\n")
        
        return {
            'metrics': metrics,
            'results': results,
            'samples': dataset[:5] if len(results) > 0 else []  # First 5 for inspection
        }