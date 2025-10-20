"""
HumanEval benchmark implementation for evaluating code generation models.
"""
import os
import re
import subprocess
import tempfile
import signal
from typing import Dict, Any, List
from .base_benchmark import BaseBenchmark
from data.prepare_data import download_humaneval


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval benchmark for code generation evaluation."""
    
    def __init__(self, tokenizer, max_new_tokens: int = 256, timeout_seconds: int = 10, 
                 use_test_execution: bool = True):
        super().__init__("HumanEval", tokenizer, max_new_tokens)
        self.timeout_seconds = timeout_seconds
        self.use_test_execution = use_test_execution
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the HumanEval dataset."""
        return list(download_humaneval())
    
    def evaluate_sample(self, model, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single HumanEval sample."""
        task_id = sample.get('task_id', 'unknown')
        prompt = sample.get('prompt', '')
        canonical_solution = sample.get('canonical_solution', '')
        test = sample.get('test', '')
        entry_point = sample.get('entry_point', '')
        
        # Generate completion
        completion = self.generate_completion(model, prompt)
        
        # Extract function definition from completion
        function_code = self._extract_function_code(completion, entry_point)
        
        # Basic checks
        has_entry_point = entry_point in completion if entry_point else False
        has_function_def = bool(re.search(r'def\s+\w+', completion))
        has_return = 'return' in completion
        
        # Execute tests if we have test code and test execution is enabled
        test_passed = False
        execution_error = None
        
        if self.use_test_execution and test and function_code:
            test_passed, execution_error = self._execute_test(function_code, test)
        
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
            'success': True
        }
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute HumanEval metrics."""
        successful_results = [r for r in results if r.get('success', True)]
        
        if not successful_results:
            return {
                'pass@1': 0.0,
                'syntax_score': 0.0,
                'entry_point_score': 0.0,
                'function_def_score': 0.0,
                'return_score': 0.0,
                'execution_success_rate': 0.0,
            }
        
        # Pass@1: Tests that actually pass
        test_passed = sum(1 for r in successful_results if r.get('test_passed', False))
        pass_at_1 = test_passed / len(successful_results)
        
        # Syntax indicators (heuristic-based)
        has_entry_point = sum(1 for r in successful_results if r.get('has_entry_point', False))
        has_function_def = sum(1 for r in successful_results if r.get('has_function_def', False))
        has_return = sum(1 for r in successful_results if r.get('has_return', False))
        
        entry_point_score = has_entry_point / len(successful_results)
        function_def_score = has_function_def / len(successful_results)
        return_score = has_return / len(successful_results)
        
        # Execution success rate (no errors)
        no_execution_errors = sum(1 for r in successful_results 
                                 if r.get('execution_error') is None)
        execution_success_rate = no_execution_errors / len(successful_results)
        
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
            'total_samples': len(successful_results),
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
            current_indent = len(line) - len(line.lstrip())
            
            if base_indent is None and line.strip().startswith('def'):
                base_indent = current_indent
                function_lines.append(line)
            elif base_indent is not None:
                if current_indent > base_indent or line.strip().startswith((' ', '\t')):
                    # Still inside function
                    function_lines.append(line)
                elif current_indent <= base_indent and line.strip() and not line.strip().startswith('#'):
                    # Function ended
                    break
                else:
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
                    error_msg = result.stderr or result.stdout or "Unknown execution error"
                    return False, error_msg
                    
            except subprocess.TimeoutExpired:
                return False, f"Execution timeout ({self.timeout_seconds}s)"
            except Exception as e:
                return False, f"Execution error: {e}"
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            return False, f"Test setup error: {e}"
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Add indentation to code."""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line 
                        for line in code.split('\n'))


# Import sys for subprocess
import sys