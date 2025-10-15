# Available Training Datasets

This document describes the datasets available for training DyLoRA-MoE and how to use them.

## Available Datasets

### 1. **code_alpaca** (Default)
- **Source**: `sahil2801/CodeAlpaca-20k`
- **Size**: ~20,000 examples
- **Format**: Instruction-following (instruction, input, output)
- **Languages**: Multilingual code examples
- **Description**: General code instruction following dataset

### 2. **mbpp** (Default)
- **Source**: `mbpp` (Google's Mostly Basic Python Problems)
- **Size**: ~1,000 examples
- **Format**: Problem descriptions with solutions
- **Languages**: Python only
- **Description**: Basic Python programming problems

### 3. **evol_instruct** ⭐ Recommended
- **Source**: `nickrosh/Evol-Instruct-Code-80k-v1`
- **Size**: ~80,000 examples (4x larger than Code Alpaca)
- **Format**: Instruction-following (instruction, output)
- **Languages**: Multilingual
- **Description**: High-quality evolved code instructions, excellent for scaling up training

### 4. **code_feedback** ⭐ Recommended
- **Source**: `m-a-p/CodeFeedback-Filtered-Instruction`
- **Size**: Variable (filtered for quality)
- **Format**: Query-answer pairs with language tags
- **Languages**: Multi-language with explicit language field
- **Description**: High-quality code instructions with feedback

### 5. **python_codes_25k**
- **Source**: `flytech/python-codes-25k`
- **Size**: ~25,000 examples
- **Format**: Instruction-following (instruction, input, output, text)
- **Languages**: Python only
- **Description**: Python-focused code examples

### 6. **python_code_instructions_18k**
- **Source**: `iamtarun/python_code_instructions_18k_alpaca`
- **Size**: ~18,000 examples
- **Format**: Alpaca-style (instruction, input, output, prompt)
- **Languages**: Python only
- **Description**: Python code instructions compatible with Code Alpaca format

### 7. **python_code_23k_sharegpt**
- **Source**: `ajibawa-2023/Python-Code-23k-ShareGPT`
- **Size**: ~23,000 examples
- **Format**: Conversational (ChatGPT-style conversations)
- **Languages**: Python only
- **Description**: Python code in conversational format

### 8. **humaneval** (Evaluation Only)
- **Source**: `openai_humaneval`
- **Size**: 164 examples
- **Format**: Programming problems with test cases
- **Languages**: Python only
- **Description**: **For evaluation only, NOT for training**

## Usage

### Basic Usage (Default)

```bash
# Uses code_alpaca and mbpp (default)
python train.py --bf16 --num_epochs 10
```

### Custom Dataset Selection

Use the `--datasets` flag with a comma-separated list:

```bash
# Use Evol-Instruct (80k) + MBPP for large-scale training
python train.py --datasets "evol_instruct,mbpp" --bf16 --num_epochs 10

# Combine multiple datasets
python train.py --datasets "code_alpaca,evol_instruct,code_feedback" --bf16 --num_epochs 10

# Python-focused training
python train.py --datasets "python_codes_25k,python_code_instructions_18k,mbpp" --bf16 --num_epochs 10
```

### Recommended Configurations

#### 🚀 Large-Scale Training (~180k examples)
```bash
python train.py --datasets "code_alpaca,evol_instruct,code_feedback,mbpp" \
  --bf16 --num_epochs 10 --num_experts 4 \
  --balance_coefficient 0.01 \
  --train_batch_size 2 --gradient_accumulation_steps 32
```

#### 🎯 Python Specialist (~60k examples)
```bash
python train.py --datasets "python_codes_25k,python_code_instructions_18k,mbpp" \
  --bf16 --num_epochs 10 --num_experts 4 \
  --balance_coefficient 0.01
```

#### ⚡ Quick Testing
```bash
python train.py --datasets "code_alpaca,mbpp" \
  --training_subset 10 --eval_subset 20 \
  --bf16 --num_epochs 3
```

## Dataset Sampling

### Interleaved Sampling (Balanced)
When using `--interleaved_sampling` with exactly 2 datasets, training samples are balanced 50/50:

```bash
python train.py --datasets "code_alpaca,mbpp" --interleaved_sampling --bf16
```

### Concatenation (Size-based)
Without `--interleaved_sampling`, datasets are concatenated and representation is proportional to size:

```bash
python train.py --datasets "evol_instruct,mbpp,code_feedback" --bf16
# Distribution will be ~80k evol_instruct, ~1k mbpp, ~Xk code_feedback
```

## Data Format Handling

The training script automatically handles different dataset formats:

- **Alpaca format**: `instruction` + `input` + `output`
- **Code Feedback format**: `query` + `answer`
- **ShareGPT format**: Concatenated `conversations`
- **MBPP format**: `text` field
- **Custom formats**: Add handling in `extract_text_from_dataset()` function in `train.py`

## Adding New Datasets

1. Add download function to `data/prepare_data.py`:
```python
def download_my_dataset(with_validation=True):
    dataset = load_dataset("username/dataset-name", split="train")
    if with_validation:
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        return DatasetDict({'train': split_dataset['train'], 'validation': split_dataset['test']})
    return dataset
```

2. Register in `AVAILABLE_DATASETS`:
```python
AVAILABLE_DATASETS = {
    ...
    'my_dataset': download_my_dataset,
}
```

3. Use it:
```bash
python train.py --datasets "my_dataset,code_alpaca" --bf16
```

## Vertex AI Cloud Training

Update `submit_full_training.py` to use custom datasets:

```python
"command": [
    "python", "train.py",
    "--datasets", "evol_instruct,mbpp,code_feedback",  # Add this line
    "--bf16",
    "--num_epochs", "10",
    # ... other args
],
```
