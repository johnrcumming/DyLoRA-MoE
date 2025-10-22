"""
Base benchmark class for evaluating models on various coding tasks.
"""
import torch
import wandb
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer
from dylo_moe.model import DyLoRA_MoE
from tqdm import tqdm


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
    
    def generate_completion(self, model, prompt: str, **generation_kwargs) -> tuple[str, dict]:
        """Generate code completion for a given prompt.
        
        Returns:
            tuple: (completion_text, metadata_dict) where metadata includes:
                - truncated: bool indicating if generation hit max_new_tokens limit
                - num_tokens: int number of tokens generated
                - prompt_tokens: int number of tokens in prompt
        """
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
        
        # Tokenize input - use larger context for code generation
        # HumanEval prompts can be long (docstrings + function signatures)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        prompt_tokens = inputs['input_ids'].shape[1]
        
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
        
        # Calculate generated tokens
        total_tokens = outputs.shape[1]
        generated_tokens = total_tokens - prompt_tokens
        
        # Check if truncated (hit max_new_tokens limit)
        # Generation is truncated if we generated exactly max_new_tokens and didn't hit EOS
        truncated = generated_tokens >= default_kwargs['max_new_tokens']
        if truncated:
            # Double-check: if last token is EOS, it wasn't actually truncated
            last_token = outputs[0, -1].item()
            if last_token == self.tokenizer.eos_token_id:
                truncated = False
        
        # Decode and extract completion
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()
        
        # Create metadata
        metadata = {
            'truncated': truncated,
            'num_tokens': generated_tokens,
            'prompt_tokens': prompt_tokens,
        }
        
        # Debug: warn if completion is empty
        if not completion:
            # Only log occasionally to avoid spam
            if not hasattr(self, '_empty_completion_count'):
                self._empty_completion_count = 0
            self._empty_completion_count += 1
            if self._empty_completion_count <= 3:  # Log first 3 occurrences
                print(f"  ⚠️  Warning: Generated empty completion (count: {self._empty_completion_count})")
                print(f"     Prompt length: {len(prompt)} chars, Generated length: {len(generated_text)} chars")
        
        model.train()
        return completion, metadata
    
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
        for i, sample in tqdm(enumerate(dataset)):
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
        
        # Print results with contextual info for execution metrics
        print(f"\n{self.name} Results:")
        for key, value in metrics.items():
            if key == 'execution_success_rate' and 'tests_run' in metrics:
                tests_run = metrics.get('tests_run', 0)
                if tests_run == 0:
                    print(f"  {key}: N/A (no tests were executed)")
                else:
                    print(f"  {key}: {value:.4f} (over {tests_run} tests)")
            else:
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