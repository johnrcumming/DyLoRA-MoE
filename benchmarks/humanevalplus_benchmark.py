"""
HumanEval+ benchmark implementation for evaluating code generation models.
HumanEval+ is an enhanced version of HumanEval with more comprehensive test cases.
"""
import os
import re
import subprocess
import tempfile
import sys
from typing import Dict, Any, List
from .base_benchmark import BaseBenchmark
from datasets import load_dataset


class HumanEvalPlusBenchmark(BaseBenchmark):
    """HumanEval+ benchmark with enhanced test coverage."""
    
    def __init__(self, tokenizer, max_new_tokens: int = 4096, timeout_seconds: int = 10,
                 use_test_execution: bool = True, use_adaptive_tokens: bool = False):
        super().__init__("HumanEval+", tokenizer, max_new_tokens, use_adaptive_tokens)
        self.timeout_seconds = timeout_seconds
        self.use_test_execution = use_test_execution
    
    def get_stop_sequences(self) -> List[str]:
        """Get HumanEval+-specific stop sequences following EvalPlus standards.
        
        HumanEval+ uses the same stop sequences as HumanEval (both part of EvalPlus).
        
        Returns:
            List of HumanEval+-specific stop strings
        """
        # Same as HumanEval - both use EvalPlus HumanEval protocol
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the HumanEval+ dataset."""
        dataset = load_dataset("evalplus/humanevalplus", split="test")
        return list(dataset)
    
    def evaluate_sample(self, model, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single HumanEval+ sample."""
        task_id = sample.get('task_id', 'unknown')
        prompt = sample.get('prompt', '')
        canonical_solution = sample.get('canonical_solution', '')
        test = sample.get('test', '')
        entry_point = sample.get('entry_point', '')
        
        # Generate completion with greedy decoding (now returns completion and metadata)
        completion, gen_metadata = self.generate_completion(model, prompt, greedy=True)
        
        # Use EvalPlus sanitization to extract function code
        function_code = self.sanitize_completion(completion, prompt, entry_point)
        
        # Debug: Check why function_code might be empty
        if self.use_test_execution and test and not function_code:
            # Log first occurrence for debugging
            if not hasattr(self, '_logged_empty_function'):
                self._logged_empty_function = True
                print(f"\n⚠️  DEBUG: Empty function_code extracted")
                print(f"  Task: {task_id}")
                print(f"  Entry point: {entry_point}")
                print(f"  Completion length: {len(completion)}")
                print(f"  Completion preview: {completion[:200] if completion else '(empty)'}")
        
        # Basic checks
        has_entry_point = entry_point in completion if entry_point else False
        has_function_def = bool(re.search(r'def\s+\w+', completion))
        has_return = 'return' in completion
        
        # Execute tests if we have test code and test execution is enabled
        test_passed = False
        execution_error = None
        test_run = False

        # Run tests if enabled and we have test code
        # Use function_code if available, otherwise try with raw completion
        if self.use_test_execution and test:
            code_to_test = function_code if function_code and function_code.strip() else completion
            if code_to_test and code_to_test.strip():  # Only run if we have non-empty code
                test_run = True
                test_passed, execution_error = self._execute_test(code_to_test, test)

        return {
            'task_id': task_id,
            'prompt': prompt,
            'completion': completion,
            'function_code': function_code,
            'canonical_solution': canonical_solution,
            'has_entry_point': has_entry_point,
            'has_function_def': has_function_def,
            'has_return': has_return,
            'test_passed': test_passed,
            'execution_error': execution_error,
            'test_run': test_run,
            'truncated': gen_metadata.get('truncated', False),
            'num_tokens': gen_metadata.get('num_tokens', 0),
            'prompt_tokens': gen_metadata.get('prompt_tokens', 0),
            'max_tokens_limit': gen_metadata.get('max_tokens_limit', self.max_new_tokens),
            'success': True
        }
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute HumanEval+ metrics."""
        successful_results = [r for r in results if r.get('success', True)]
        
        if not successful_results:
            return {
                'pass@1': 0.0,
                'syntax_score': 0.0,
                'entry_point_score': 0.0,
                'function_def_score': 0.0,
                'return_score': 0.0,
                'execution_success_rate': 0.0,
                'truncation_rate': 0.0,
                'avg_tokens_generated': 0.0,
                'avg_prompt_tokens': 0.0,
            }
        
        # Pass@1: Tests that actually pass (stricter in HumanEval+ due to more tests)
        test_passed = sum(1 for r in successful_results if r.get('test_passed', False))
        pass_at_1 = test_passed / len(successful_results)
        
        # Syntax indicators (heuristic-based)
        has_entry_point = sum(1 for r in successful_results if r.get('has_entry_point', False))
        has_function_def = sum(1 for r in successful_results if r.get('has_function_def', False))
        has_return = sum(1 for r in successful_results if r.get('has_return', False))
        
        entry_point_score = has_entry_point / len(successful_results)
        function_def_score = has_function_def / len(successful_results)
        return_score = has_return / len(successful_results)
        
        # Execution success rate (no errors) — computed only over samples where tests were run
        tests_run = sum(1 for r in successful_results if r.get('test_run'))
        if tests_run > 0:
            no_execution_errors = sum(1 for r in successful_results if r.get('test_run') and r.get('execution_error') is None)
            execution_success_rate = no_execution_errors / tests_run
        else:
            execution_success_rate = 0.0

        # Truncation statistics
        truncated_count = sum(1 for r in successful_results if r.get('truncated', False))
        truncation_rate = truncated_count / len(successful_results)
        
        # Token statistics
        total_tokens_generated = sum(r.get('num_tokens', 0) for r in successful_results)
        avg_tokens_generated = total_tokens_generated / len(successful_results)
        
        total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in successful_results)
        avg_prompt_tokens = total_prompt_tokens / len(successful_results)
        
        # Max token limit statistics (now fixed, not adaptive)
        token_limits = [r.get('max_tokens_limit', self.max_new_tokens) for r in successful_results]
        max_token_limit = max(token_limits) if token_limits else self.max_new_tokens
        min_token_limit = min(token_limits) if token_limits else self.max_new_tokens
        avg_token_limit = sum(token_limits) / len(token_limits) if token_limits else self.max_new_tokens
        
        # Log truncation warning if rate is significant
        if truncation_rate > 0.05:  # More than 5% truncated
            print(f"\n⚠️  Truncation Alert: {truncation_rate:.1%} of completions hit max_new_tokens limit")
            print(f"   Consider increasing max_new_tokens (current: {self.max_new_tokens})")
            # Show examples of truncated tasks
            truncated_tasks = [r.get('task_id', 'unknown') for r in successful_results if r.get('truncated', False)]
            print(f"   Truncated tasks: {truncated_tasks[:5]}" + (" ..." if len(truncated_tasks) > 5 else ""))

        # Combined syntax score
        syntax_score = (entry_point_score + function_def_score + return_score) / 3
        
        return {
            'pass@1': pass_at_1,
            'syntax_score': syntax_score,
            'entry_point_score': entry_point_score,
            'function_def_score': function_def_score,
            'return_score': return_score,
            'execution_success_rate': execution_success_rate,
            'tests_passed': test_passed,
            'tests_run': tests_run,
            'total_samples': len(successful_results),
            'truncation_rate': truncation_rate,
            'truncated_count': truncated_count,
            'avg_tokens_generated': avg_tokens_generated,
            'avg_prompt_tokens': avg_prompt_tokens,
            'max_token_limit': max_token_limit,
            'min_token_limit': min_token_limit,
            'avg_token_limit': avg_token_limit,
        }
    
    def _extract_function_code(self, completion: str, entry_point: str) -> str:
        """Extract the complete function definition from completion."""
        if not entry_point:
            return completion
        
        # Try to find the function definition
        # Look for 'def entry_point(' pattern
        pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\):'
        match = re.search(pattern, completion)
        
        if not match:
            # Fallback: return the whole completion
            return completion
        
        start_pos = match.start()
        
        # Find the end of the function by looking for the next 'def' or end of string
        lines = completion[start_pos:].split('\n')
        function_lines = []
        in_function = True
        base_indent = None
        
        for line in lines:
            if not line.strip():  # Empty line
                if in_function:
                    function_lines.append(line)
                continue
            
            # Determine indentation
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            # First line with content sets the base indent
            if base_indent is None:
                base_indent = indent
                function_lines.append(line)
                continue
            
            # If we hit another function at same or less indentation, stop
            if stripped.startswith('def ') and indent <= base_indent:
                break
            
            # If indentation is less than base and line has content, we're done
            if indent < base_indent:
                break
            
            # Otherwise, this line is part of the function
            if in_function:
                function_lines.append(line)
        
        return '\n'.join(function_lines)
    
    def _execute_test(self, function_code: str, test_code: str) -> tuple[bool, str]:
        """Execute the test code with the function and return (success, error_message)."""
        try:
            # Create complete test program
            full_code = f"""
import sys
import traceback

{function_code}

try:
{self._indent_code(test_code, 4)}
    print("TEST_PASSED")
except Exception as e:
    print(f"TEST_FAILED: {{e}}")
    traceback.print_exc()
"""
            
            # Write to temporary file and execute
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
            
            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                
                # Check if test passed
                if "TEST_PASSED" in result.stdout:
                    return True, None
                else:
                    # Extract error message
                    error_msg = result.stdout + result.stderr
                    return False, error_msg[:200]  # Truncate long errors
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {self.timeout_seconds}s"
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Add indentation to code."""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line 
                        for line in code.split('\n'))
