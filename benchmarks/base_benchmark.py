"""
Base benchmark class for evaluating models on various coding tasks.
Integrated with EvalPlus framework for rigorous evaluation.
"""
import torch
import wandb
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Tuple
from transformers import AutoTokenizer
from dylo_moe.model import DyLoRA_MoE
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

# Import EvalPlus sanitization utilities
from evalplus.sanitize import sanitize


class AsyncTestExecutor:
    """Async test executor for parallel test execution during benchmarking.
    
    This class manages a thread pool for running test subprocesses in parallel
    while the GPU continues generating code for the next samples. This improves
    GPU utilization from ~15-30% (sequential) to 60-80%+ (pipelined).
    
    Usage:
        executor = AsyncTestExecutor(max_workers=4)
        future = executor.submit_test(test_func, code, test_spec, sample_index)
        # ... do GPU work ...
        result = executor.get_result(future)
        executor.shutdown()
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize async test executor.
        
        Args:
            max_workers: Max concurrent test processes. Default is cpu_count() // 2
        """
        if max_workers is None:
            max_workers = max(1, os.cpu_count() // 2)
        
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.submitted_count = 0
        self.completed_count = 0
        
    def submit_test(
        self, 
        test_func: Callable, 
        *args,
        **kwargs
    ) -> Tuple[Future, int]:
        """Submit a test for async execution.
        
        Args:
            test_func: The test function to execute (e.g., _execute_test)
            *args: Positional arguments for test_func
            **kwargs: Keyword arguments for test_func
            
        Returns:
            Tuple of (Future object, submission_index)
        """
        submission_index = self.submitted_count
        self.submitted_count += 1
        future = self.executor.submit(test_func, *args, **kwargs)
        return future, submission_index
    
    def get_result(self, future: Future, timeout: Optional[float] = None) -> Any:
        """Get result from a completed future.
        
        Args:
            future: Future object from submit_test
            timeout: Max seconds to wait (None = wait forever)
            
        Returns:
            Result from test function
            
        Raises:
            TimeoutError: If timeout is exceeded
            Exception: Any exception raised by test function
        """
        try:
            result = future.result(timeout=timeout)
            self.completed_count += 1
            return result
        except Exception as e:
            self.completed_count += 1
            raise e
    
    def get_pending_count(self) -> int:
        """Get number of tests currently pending."""
        return self.submitted_count - self.completed_count
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor and cleanup resources.
        
        Args:
            wait: If True, wait for all pending tests to complete
        """
        self.executor.shutdown(wait=wait)


def get_adaptive_max_tokens(prompt: str, benchmark_name: str, 
                            base_limit: int = None) -> int:
    """
    Calculate adaptive max_new_tokens based on prompt complexity.
    DEPRECATED: Now using fixed maximum limits to avoid truncation.
    
    Args:
        prompt: The input prompt text
        benchmark_name: Name of benchmark (humaneval, humanevalplus, mbpp)
        base_limit: Override default base limit for the benchmark
    
    Returns:
        Fixed max token limit (no adaptation)
    """
    # Fixed high limits to eliminate truncation
    # Based on empirical data showing high truncation rates
    default_limits = {
        'humaneval': 4096,      # Was experiencing 34% truncation even at 2048
        'humanevalplus': 4096,  # Was experiencing 38% truncation even at 2048
        'mbpp': 4096            # Was experiencing 60% truncation even at 3072
    }
    
    # Use provided base_limit or fallback to benchmark default
    if base_limit is not None:
        return base_limit
    else:
        return default_limits.get(benchmark_name.lower(), 4096)


class BaseBenchmark(ABC):
    """Abstract base class for model benchmarks."""
    
    def __init__(self, name: str, tokenizer: AutoTokenizer, max_new_tokens: int = 256, 
                 use_adaptive_tokens: bool = True, use_async_tests: bool = False,
                 max_concurrent_tests: Optional[int] = None):
        """Initialize benchmark.
        
        Args:
            name: Benchmark name
            tokenizer: Model tokenizer
            max_new_tokens: Max tokens to generate
            use_adaptive_tokens: Legacy parameter (now deprecated)
            use_async_tests: Enable async test execution for better GPU utilization
            max_concurrent_tests: Max concurrent test processes (default: cpu_count() // 2)
        """
        self.name = name
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.use_adaptive_tokens = use_adaptive_tokens
        self.use_async_tests = use_async_tests
        self.max_concurrent_tests = max_concurrent_tests
        self._test_executor = None
        
    def _get_test_executor(self) -> Optional[AsyncTestExecutor]:
        """Get or create the async test executor (lazy initialization)."""
        if self.use_async_tests and self._test_executor is None:
            self._test_executor = AsyncTestExecutor(max_workers=self.max_concurrent_tests)
        return self._test_executor
    
    def _cleanup_test_executor(self):
        """Cleanup async test executor if it exists."""
        if self._test_executor is not None:
            self._test_executor.shutdown(wait=True)
            self._test_executor = None
        
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
    
    def get_stop_sequences(self) -> List[str]:
        """Get benchmark-specific stop sequences following EvalPlus standards.
        
        Returns:
            List of stop strings for this benchmark. Override in subclasses.
        """
        # Default stop sequences (HumanEval-style)
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    
    def generate_completion(self, model, prompt: str, greedy: bool = True, **generation_kwargs) -> tuple[str, dict]:
        """Generate code completion for a given prompt.
        
        Args:
            model: The model to generate with
            prompt: Input prompt text
            greedy: If True, use greedy decoding (temperature=0, no sampling). Default True.
            **generation_kwargs: Additional generation parameters (override defaults)
        
        Returns:
            tuple: (completion_text, metadata_dict) where metadata includes:
                - truncated: bool indicating if generation hit max_new_tokens limit
                - num_tokens: int number of tokens generated
                - prompt_tokens: int number of tokens in prompt
                - adaptive_limit: int the max_new_tokens used (if adaptive)
        """
        # Use max token limit directly (no dynamic adjustment)
        # Adaptive tokens disabled - just use the maximum to avoid truncation
        max_tokens = generation_kwargs.get('max_new_tokens', self.max_new_tokens)
        
        # Default generation parameters
        if greedy:
            # Greedy decoding for deterministic, reproducible results (EvalPlus standard)
            default_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': 0.0,
                'do_sample': False,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            # Add benchmark-specific stop sequences following EvalPlus protocol
            # These prevent the model from generating beyond the required function
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                stop_strings = self.get_stop_sequences()
                default_kwargs['stop_strings'] = stop_strings
                # IMPORTANT: Must pass tokenizer when using stop_strings
                default_kwargs['tokenizer'] = self.tokenizer
        else:
            # Sampling mode (legacy)
            default_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': 0.2,
                'do_sample': True,
                'top_p': 0.95,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
        default_kwargs.update(generation_kwargs)
        
        # Tokenize input - NO TRUNCATION for prompts!
        # Code generation prompts should never be cut off
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
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
            'max_tokens_limit': max_tokens,  # Fixed limit used (no adaptation)
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
    
    def sanitize_completion(self, completion: str, prompt: str, entry_point: Optional[str] = None) -> str:
        """Sanitize generated code using EvalPlus tree-sitter extraction.
        
        Args:
            completion: Raw generated code
            prompt: Original prompt (used to reconstruct full solution)
            entry_point: Function name to extract dependencies for
            
        Returns:
            Sanitized code with only relevant function definitions
        """
        # Combine prompt + completion for full context
        full_solution = prompt + "\n" + completion
        
        # Use EvalPlus sanitization
        try:
            sanitized = sanitize(code=full_solution, entrypoint=entry_point)
            return sanitized
        except Exception as e:
            # Fallback to returning raw completion if sanitization fails
            print(f"  ⚠️  Sanitization failed: {e}, returning raw completion")
            return completion
    
    def run_benchmark(self, model, max_samples: Optional[int] = None, 
                     log_to_wandb: bool = False, prefix: str = "") -> Dict[str, Any]:
        """Run the complete benchmark on a model."""
        print(f"\n{'='*80}")
        print(f"Running {self.name} Benchmark")
        if prefix:
            print(f"Model: {prefix}")
        print(f"{'='*80}")
        print(f"Max tokens for generation: {self.max_new_tokens}")
        print(f"Adaptive tokens: disabled (using fixed maximum to eliminate truncation)")
        print(f"Prompt truncation: disabled (full prompts preserved)")
        
        # Load dataset
        dataset = self.load_dataset()
        if max_samples:
            dataset = dataset[:max_samples]
        
        print(f"Evaluating on {len(dataset)} {self.name} problems...")
        
        # Check if async mode is enabled
        if self.use_async_tests:
            print(f"  ℹ️  Async test execution enabled (max workers: {self.max_concurrent_tests or 'auto'})")
            results = self._run_benchmark_async(model, dataset)
        else:
            results = self._run_benchmark_sync(model, dataset)
        
        # Cleanup test executor if it exists
        self._cleanup_test_executor()
        
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
    
    def _run_benchmark_sync(self, model, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run benchmark synchronously (original sequential implementation)."""
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
        return results
    
    def _run_benchmark_async(self, model, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run benchmark with async test execution for improved GPU utilization.
        
        Pipeline:
        1. Generate completion on GPU (blocking)
        2. Submit test execution to ThreadPoolExecutor (non-blocking)
        3. Continue to next sample immediately
        4. After all generations complete, collect all test results
        
        NOTE: Test execution runs on CPU (subprocess), not GPU.
        This overlaps GPU generation with CPU testing for better utilization.
        """
        executor = self._get_test_executor()
        if executor is None:
            print("  ⚠️  Failed to initialize async executor, falling back to sync mode")
            return self._run_benchmark_sync(model, dataset)
        
        # Check if benchmark supports async methods
        has_async_support = (hasattr(self, 'generate_completion_for_async') and 
                            hasattr(self, 'execute_tests_async'))
        
        if not has_async_support:
            print("  ⚠️  Benchmark does not support async execution, falling back to sync mode")
            return self._run_benchmark_sync(model, dataset)
        
        print(f"  ✓ Async execution enabled: max_workers={self.max_concurrent_tests or 'auto'}")
        
        # Phase 1: Generate completions and submit tests
        print(f"  Phase 1: Generating completions and submitting tests...")
        pending_tests = []  # List of (Future, sample_index, generation_result)
        tests_submitted = 0
        
        for i, sample in tqdm(enumerate(dataset), desc="Generating", total=len(dataset)):
            try:
                # Generate completion (GPU operation - blocking)
                generation_result = self.generate_completion_for_async(model, sample)
                generation_result['sample_index'] = i
                
                # Submit test execution to executor (non-blocking)
                # Check if benchmark has test execution enabled (default to True for async)
                use_tests = getattr(self, 'use_test_execution', True)
                if use_tests:
                    future, _ = executor.submit_test(self.execute_tests_async, generation_result)
                    pending_tests.append((future, i, None))
                    tests_submitted += 1
                else:
                    # No test execution, store result directly
                    pending_tests.append((None, i, generation_result))
                
                # Log progress periodically
                if (i + 1) % 20 == 0:
                    pending_count = executor.get_pending_count()
                    print(f"  Generated {i + 1}/{len(dataset)}, {tests_submitted} tests submitted, {pending_count} tests pending")
                
            except Exception as e:
                print(f"  ⚠️  Error generating sample {i}: {e}")
                pending_tests.append((None, i, {
                    'sample_index': i,
                    'task_id': sample.get('task_id', f'{self.name}_{i}'),
                    'success': False,
                    'error': str(e)
                }))
        
        print(f"  ✓ Phase 1 complete: {len(dataset)} generations, {tests_submitted} tests submitted")
        print(f"  Pending tests at end of generation: {executor.get_pending_count()}")
        
        # Phase 2: Collect test results
        print(f"  Phase 2: Collecting results from {len(pending_tests)} samples...")
        results = []
        completed = 0
        
        for future, idx, generation_result in tqdm(pending_tests, desc="Testing"):
            if future is None:
                # No async test, use stored generation result
                results.append(generation_result)
            else:
                # Wait for async test to complete
                try:
                    result = executor.get_result(future, timeout=60)
                    results.append(result)
                except Exception as e:
                    print(f"  ⚠️  Error collecting result for sample {idx}: {e}")
                    results.append({
                        'sample_index': idx,
                        'task_id': f'{self.name}_{idx}',
                        'success': False,
                        'error': str(e)
                    })
            
            completed += 1
            if completed % 10 == 0:
                pending = executor.get_pending_count()
                print(f"  Completed {completed}/{len(pending_tests)}, {pending} tests still running...")
        
        return results
        
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