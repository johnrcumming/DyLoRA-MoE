# HumanEval+ Baseline Pass@1 Inflation Issue

## Problem Summary

Training logs show unrealistic HumanEval+ pass@1 scores:
- **Baseline**: 99.39% (published CodeGemma-2B: 20.7%)
- **Epoch 1**: 79.27% 
- **Epochs 2-3 & Final**: 100.00%

## Root Cause Analysis

### Issue 1: Aggressive Repetition Penalty at Baseline

**Location**: `benchmarks/base_benchmark.py` line 221

**Problem**:
```python
'repetition_penalty': 1.1,  # Penalize repetition to prevent verbosity
```

This penalty was intended to reduce verbosity during training evaluations (epochs 1-3), but it was also applied to **baseline** evaluation, causing:

- **Baseline avg tokens**: 8.8 tokens (extremely short)
- **Baseline syntax scores**: All near 0% (syntax: 1.42%, entry_point: 0.61%, function_def: 0.00%)
- **Inflated pass@1**: 99.39% despite minimal code generation

### Issue 2: Prompt + Completion Concatenation

**Location**: `benchmarks/base_benchmark.py` line 313 (in `sanitize_completion`)

**Code**:
```python
def sanitize_completion(self, completion: str, prompt: str, entry_point: Optional[str] = None) -> str:
    # Combine prompt + completion for full context
    full_solution = prompt + "\n" + completion
    
    # Use EvalPlus sanitization
    try:
        sanitized = sanitize(code=full_solution, entrypoint=entry_point)
        return sanitized
```

**Why This Causes Inflation**:

1. **HumanEval+ prompts** include full function signatures with docstrings:
   ```python
   def has_close_elements(numbers: List[float], threshold: float) -> bool:
       """ Check if in given list of numbers, are any two numbers closer to each other than
       given threshold.
       >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
       False
       """
   ```

2. **With repetition_penalty=1.1**, model generates only ~8.8 tokens (incomplete/minimal code)

3. **Sanitizer extracts** function from `prompt + completion`, which includes the full signature from the prompt

4. **Test execution** runs with the function signature (which may have implicit `pass` or return `None`)

5. **Tests pass** because:
   - Some tests are lenient about return types
   - Function signature alone is "syntactically valid"
   - EvalPlus might accept partial implementations in some cases

### Evidence from Logs

**Baseline (repetition_penalty=1.1)**:
```
pass@1: 0.9939 (163/164)
avg_tokens_generated: 8.8354
syntax_score: 0.0142 (1.42%)
entry_point_score: 0.0061 (0.61%)
function_def_score: 0.0000 (0%)
```

**Epoch 2-3 (after training, still with repetition_penalty=1.1)**:
```
pass@1: 1.0000 (164/164)
avg_tokens_generated: 1024.0000 (hitting max limit)
syntax_score: 0.0000 (0%)
entry_point_score: 0.0000 (0%)
truncation_rate: 1.0000 (100%)
```

All structure scores are **0%**, yet pass@1 is **100%**. This is impossible for valid Python code.

## Fix Applied

**File**: `benchmarks/base_benchmark.py` (lines 218-227)

**Change**:
```python
# BEFORE (incorrect):
default_kwargs = {
    'max_new_tokens': max_tokens,
    'temperature': 0.0,
    'do_sample': False,
    'repetition_penalty': 1.1,  # ❌ Too aggressive for baseline
    ...
}

# AFTER (fixed):
default_kwargs = {
    'max_new_tokens': max_tokens,
    'temperature': 0.0,
    'do_sample': False,
    # ✅ No repetition_penalty - let models generate naturally
    # repetition_penalty=1.1 was causing baseline to generate only ~8.8 tokens,
    # leading to inflated pass@1 scores (99.39%) by relying on function 
    # signatures from prompts rather than complete implementations.
    ...
}
```

**Additional Debug Logging**:
Added sample logging in `humanevalplus_benchmark.py` (lines 53-72) to inspect:
- Prompt length
- Completion length and token count
- Extracted function code
- Preview of actual generated content

## Expected Behavior After Fix

**Baseline should now show**:
- **Lower pass@1** (~15-25%, closer to published 20.7%)
- **Higher token counts** (200-500 tokens for complete implementations)
- **Non-zero structure scores** (syntax, entry_point, function_def)
- **Realistic progression** through training epochs

**Training epochs should show**:
- **Gradual improvement** in pass@1 from baseline
- **Balanced token generation** (not hitting 1024 limit every time)
- **Improved structure scores** as model learns proper code formatting

## Testing Recommendations

1. **Re-run baseline** without repetition_penalty to get realistic score
2. **Compare token counts**: Should see 200-500+ tokens at baseline vs 8.8
3. **Inspect sample outputs**: Use debug logging to verify actual code quality
4. **Validate structure scores**: Should correlate with pass@1 (both improve together)

## Long-Term Considerations

### Keep prompt+completion concatenation?

The current approach of concatenating `prompt + completion` before sanitization is **intentional** for EvalPlus compatibility, as their framework expects complete solutions. However, this means:

**Pros**:
- Compatible with EvalPlus evaluation protocol
- Handles cases where completion references prompt context
- Standard practice for HumanEval/HumanEval+ evaluation

**Cons**:
- Can mask poor generations when model outputs minimal code
- Makes it harder to distinguish "model completed the function" vs "prompt included signature"

**Recommendation**: Keep concatenation but ensure:
1. No repetition penalties distort baseline generation
2. Debug logging shows actual completion content
3. Structure scores validate that meaningful code was generated (not just signatures)

## References

- **CodeGemma-2B HumanEval+ published score**: 20.7% (from model card)
- **EvalPlus framework**: https://github.com/evalplus/evalplus
- **Issue manifested in**: Training run 2025-10-28 17:02:10 - 2025-10-29 07:35:27
