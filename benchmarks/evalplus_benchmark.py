"""
EvalPlus-based benchmark implementation.
Uses the official EvalPlus framework for both code generation and evaluation.
"""
import os
import json
import tempfile
import shutil
from typing import Dict, Any, List, Optional
from .base_benchmark import BaseBenchmark


class EvalPlusBenchmark(BaseBenchmark):
    """Benchmark using native EvalPlus framework for generation and evaluation."""
    
    def __init__(self, tokenizer, model_name: Optional[str] = None, model_path: Optional[str] = None,
                 dataset: str = "humaneval", max_new_tokens: int = 4096, greedy: bool = True,
                 backend: str = "hf", root: str = "evalplus_results",
                 force_base_prompt: bool = False, mini: bool = False):
        """Initialize EvalPlus benchmark.
        
        Args:
            tokenizer: HuggingFace tokenizer
            model_name: Model name for HuggingFace (for base models)
            model_path: Path to pre-loaded model directory (for trained models)
            dataset: 'humaneval' or 'mbpp'
            max_new_tokens: Max tokens to generate
            greedy: Use greedy decoding (temperature=0)
            backend: EvalPlus backend ('hf', 'vllm', etc.)
            root: Root directory for results
            force_base_prompt: Force base model prompts (not chat)
            mini: Use mini version of dataset (faster)
        """
        super().__init__(f"EvalPlus-{dataset}", tokenizer, max_new_tokens, 
                        use_adaptive_tokens=False, use_async_tests=False)
        self.model_name = model_name
        self.model_path = model_path
        self.dataset = dataset
        self.greedy = greedy
        self.backend = backend
        self.root = root
        self.force_base_prompt = force_base_prompt
        self.mini = mini
        self.samples_path = None
        self.temp_model_dir = None
        
    def get_stop_sequences(self) -> List[str]:
        """EvalPlus handles stop sequences internally."""
        return []
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset via EvalPlus."""
        if self.dataset == "humaneval":
            from evalplus.data import get_human_eval_plus
            problems = get_human_eval_plus(mini=self.mini)
        elif self.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus
            problems = get_mbpp_plus(mini=self.mini)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        # Convert to list format
        return [{"task_id": k, **v} for k, v in problems.items()]
    
    def evaluate_sample(self, model, sample: Dict[str, Any]) -> Dict[str, Any]:
        """EvalPlus handles evaluation via batch processing, not per-sample."""
        raise NotImplementedError("Use run_benchmark() instead")
    
    def run_benchmark(self, model, max_samples: Optional[int] = None,
                     log_to_wandb: bool = False, prefix: str = "") -> Dict[str, Any]:
        """Run benchmark using EvalPlus framework.
        
        This bypasses the parent class's per-sample evaluation and uses
        EvalPlus's native batch generation + evaluation pipeline.
        
        For pre-loaded models (like W&B artifacts), we save them to a temp directory
        and pass that path to EvalPlus.
        """
        from evalplus.codegen import run_codegen
        from evalplus.evaluate import evaluate
        
        print(f"\n{'='*80}")
        print(f"Running {self.name} Benchmark with EvalPlus Framework")
        print(f"{'='*80}\n")
        
        # Determine the model path for EvalPlus
        evalplus_model_path = None
        
        if self.model_path:
            # Pre-loaded DyLoRA-MoE model - ALWAYS merge PEFT/LoRA adapters
            print(f"  Detected pre-loaded model from: {self.model_path}")
            print(f"  ⚠️  Merging LoRA adapters for evaluation (DyLoRA-MoE always uses PEFT)...")
            
            try:
                from peft import PeftModel
                
                # Create temp directory for merged model
                self.temp_model_dir = tempfile.mkdtemp(prefix="evalplus_merged_model_")
                
                # Check if model has merge_and_unload method (PEFT model)
                if hasattr(model, 'merge_and_unload'):
                    print(f"  🔍 Model type: {type(model).__name__} (has merge_and_unload)")
                    merged_model = model.merge_and_unload()
                elif isinstance(model, PeftModel):
                    print(f"  🔍 Model type: PeftModel (using merge_and_unload)")
                    merged_model = model.merge_and_unload()
                else:
                    # Fallback: model might already be merged or is base model
                    print(f"  ⚠️  Model type: {type(model).__name__} (no merge_and_unload, using as-is)")
                    merged_model = model
                
                # Save merged model
                merged_model.save_pretrained(self.temp_model_dir)
                self.tokenizer.save_pretrained(self.temp_model_dir)
                
                evalplus_model_path = self.temp_model_dir
                print(f"  ✓ Model saved to: {self.temp_model_dir}")
                
            except Exception as e:
                print(f"  ❌ Failed to merge/save model: {e}")
                import traceback
                traceback.print_exc()
                print(f"  ⚠️  Falling back to original path (may not include LoRA weights): {self.model_path}")
                evalplus_model_path = self.model_path
        elif self.model_name:
            # Base model - EvalPlus will load it
            evalplus_model_path = self.model_name
            print(f"  Model: {self.model_name}")
        else:
            # Need to save the model temporarily
            print(f"  Saving pre-loaded model to temporary directory...")
            
            # Check if this is a PEFT model
            try:
                from peft import PeftModel
                is_peft = isinstance(model, PeftModel)
            except:
                is_peft = False
            
            try:
                self.temp_model_dir = tempfile.mkdtemp(prefix="evalplus_model_")
                
                if is_peft:
                    print(f"  ⚠️  PEFT model detected - merging LoRA adapters...")
                    merged_model = model.merge_and_unload()
                    merged_model.save_pretrained(self.temp_model_dir)
                else:
                    model.save_pretrained(self.temp_model_dir)
                
                self.tokenizer.save_pretrained(self.temp_model_dir)
                evalplus_model_path = self.temp_model_dir
                print(f"  ✓ Model saved to: {self.temp_model_dir}")
            except Exception as e:
                print(f"  ❌ Failed to save model: {e}")
                # Fall back to model name if available
                if hasattr(model, 'name_or_path'):
                    evalplus_model_path = model.name_or_path
                    print(f"  ⚠️  Falling back to model name: {evalplus_model_path}")
                else:
                    raise ValueError("Cannot determine model path for EvalPlus")
        
        print(f"  Dataset: {self.dataset}")
        print(f"  Backend: {self.backend}")
        print(f"  Greedy: {self.greedy}")
        print(f"  Max tokens: {self.max_new_tokens}")
        print(f"  Mini: {self.mini}")
        
        # Step 1: Generate code samples using EvalPlus
        print(f"\n📝 Generating code samples...")
        
        # Determine id_range for max_samples
        id_range = None
        if max_samples:
            dataset_obj = self.load_dataset()
            total_samples = len(dataset_obj)
            if max_samples < total_samples:
                id_range = [0, max_samples - 1]
                print(f"   Limiting to first {max_samples} samples")
        
        try:
            samples_path = run_codegen(
                model=evalplus_model_path,
                dataset=self.dataset,
                root=self.root,
                bs=1,  # Batch size
                n_samples=1,  # Number of samples per task (greedy=1)
                temperature=0.0 if self.greedy else 0.8,
                resume=True,  # Resume if samples exist
                greedy=self.greedy,
                id_range=id_range,
                backend=self.backend,
                force_base_prompt=self.force_base_prompt,
                dtype="bfloat16",
                trust_remote_code=False,
            )
            
            print(f"✓ Code generation complete: {samples_path}")
            self.samples_path = samples_path
            
        except Exception as e:
            print(f"❌ Code generation failed: {e}")
            return {
                'metrics': {
                    'pass@1': 0.0,
                    'base_pass@1': 0.0,
                    'plus_pass@1': 0.0,
                    'error': str(e)
                },
                'samples_path': None
            }
        
        # Step 2: Evaluate generated code using EvalPlus
        print(f"\n🧪 Evaluating generated code...")
        
        try:
            # EvalPlus evaluate() writes results to a file and returns None
            # We need to read the results file after evaluation
            # EvalPlus names the file: <samples_base>_eval_results.json
            # samples_path is like: evalplus_results/humaneval/google--codegemma-2b_hf_temp_0.0.jsonl
            # result_path should be: evalplus_results/humaneval/google--codegemma-2b_hf_temp_0.0_eval_results.json
            if samples_path.endswith('.jsonl'):
                result_path = samples_path[:-6] + '_eval_results.json'
            else:
                result_path = samples_path + '_eval_results.json'
            
            evaluate(
                dataset=self.dataset,
                samples=samples_path,
                base_only=False,  # Include plus tests
                parallel=None,  # Use default parallelism
                i_just_wanna_run=False,  # Use cached results if available
                test_details=False,  # Don't need detailed test results
                mini=self.mini,
            )
            
            print(f"✓ Evaluation complete: {result_path}")
            
            # Read results
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    eval_results = json.load(f)
                
                # Extract metrics from EvalPlus results
                metrics = self._parse_evalplus_results(eval_results)
                
                # Log to wandb if requested
                if log_to_wandb:
                    try:
                        import wandb
                        wandb.log({f"{prefix}/{self.dataset}/{k}": v 
                                  for k, v in metrics.items()})
                    except Exception as e:
                        print(f"⚠️  W&B logging failed: {e}")
                
                return {
                    'metrics': metrics,
                    'samples_path': samples_path,
                    'results_path': result_path
                }
            else:
                print(f"⚠️  Results file not found: {result_path}")
                return {
                    'metrics': {'error': 'Results file not found'},
                    'samples_path': samples_path
                }
                
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {
                'metrics': {
                    'pass@1': 0.0,
                    'error': str(e)
                },
                'samples_path': samples_path
            }
    
    def _parse_evalplus_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse EvalPlus evaluation results into standard metrics.
        
        EvalPlus results format:
        {
            "date": "...",
            "hash": "...",
            "eval": {
                "task_id": {
                    "base": [pass/fail per sample],
                    "plus": [pass/fail per sample]
                }
            }
        }
        """
        eval_data = results.get('eval', {})
        
        if not eval_data:
            return {'pass@1': 0.0, 'base_pass@1': 0.0, 'plus_pass@1': 0.0}
        
        # Count passes
        total_tasks = len(eval_data)
        base_passes = 0
        plus_passes = 0
        
        for task_id, task_results in eval_data.items():
            # Each task has 'base' and 'plus' lists of pass/fail
            base_results = task_results.get('base', [])
            plus_results = task_results.get('plus', [])
            
            # For greedy (n=1), just check if the single sample passed
            if base_results and base_results[0]:
                base_passes += 1
            if plus_results and plus_results[0]:
                plus_passes += 1
        
        # Calculate pass@1 rates
        base_pass_rate = (base_passes / total_tasks * 100) if total_tasks > 0 else 0.0
        plus_pass_rate = (plus_passes / total_tasks * 100) if total_tasks > 0 else 0.0
        
        # Overall pass@1 is typically the 'plus' rate (more rigorous)
        metrics = {
            'pass@1': plus_pass_rate,
            'base_pass@1': base_pass_rate,
            'plus_pass@1': plus_pass_rate,
            'total_tasks': total_tasks,
            'base_passed': base_passes,
            'plus_passed': plus_passes,
        }
        
        print(f"\n{'='*80}")
        print(f"EvalPlus Results Summary:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Base tests pass@1: {base_pass_rate:.2f}% ({base_passes}/{total_tasks})")
        print(f"  Plus tests pass@1: {plus_pass_rate:.2f}% ({plus_passes}/{total_tasks})")
        print(f"{'='*80}\n")
        
        return metrics
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Not used - EvalPlus handles metrics computation."""
        raise NotImplementedError("EvalPlus computes metrics internally")
    
    def __del__(self):
        """Cleanup temporary model directory if it was created."""
        if self.temp_model_dir and os.path.exists(self.temp_model_dir):
            try:
                shutil.rmtree(self.temp_model_dir)
                print(f"🗑️  Cleaned up temporary model directory: {self.temp_model_dir}")
            except Exception as e:
                print(f"⚠️  Failed to cleanup temp directory: {e}")
