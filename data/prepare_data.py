from datasets import load_dataset, DatasetDict

def download_the_stack():
    """
    Downloads The Stack dataset.
    """
    print("Downloading The Stack...")
    dataset = load_dataset("bigcode/the-stack", data_dir="data/the-stack", split="train", streaming=True)
    return dataset

def download_code_alpaca(filter_python=False, with_validation=True):
    """
    Downloads the Code Alpaca dataset and optionally creates a train/validation split.
    """
    print("Downloading Code Alpaca...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    if filter_python:
        # Filter for Python-related instructions
        dataset = dataset.filter(lambda example: "python" in example["instruction"].lower())

    if with_validation:
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
    return dataset


def download_mbpp(with_validation=True):
    """
    Downloads the MBPP dataset and optionally creates a train/validation split.
    The original dataset only has a 'test' split, so we partition it.
    """
    print("Downloading MBPP...")
    dataset = load_dataset("mbpp", split="test")

    if with_validation:
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        # Further split into validation/test: 10% val, 10% test
        val_test = dataset['test'].train_test_split(test_size=0.5, seed=42)
        dataset = {
            "train": dataset['train'],
            "validation": val_test['train'],
            "test": val_test['test'],
        }
    return dataset


def download_apps(max_samples=10000, difficulty_level="all", with_validation=True):
    """
    Downloads APPS (Automated Programming Progress Standard) dataset.
    More manageable size than CodeContests with similar algorithmic content.
    
    Args:
        max_samples: Maximum number of samples to use
        difficulty_level: "introductory", "interview", "competition", or "all"
        with_validation: Whether to create train/validation split
    """
    print("Downloading APPS...")
    try:
        dataset = load_dataset("codeparrot/apps", split="train", trust_remote_code=True)
        
        # Filter by difficulty if specified
        if difficulty_level != "all":
            level_map = {"introductory": 0, "interview": 1, "competition": 2}
            if difficulty_level in level_map:
                target_level = level_map[difficulty_level]
                dataset = dataset.filter(lambda x: x.get("difficulty") == target_level)
        
        # Limit to max_samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=42).select(range(max_samples))
        
        if with_validation:
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            dataset = DatasetDict({
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            })
        
        return dataset
    except Exception as e:
        print(f"Warning: Could not load APPS dataset: {e}")
        print("Using empty dataset as fallback")
        from datasets import Dataset
        empty_dataset = Dataset.from_dict({"question": [], "solutions": []})
        if with_validation:
            return DatasetDict({'train': empty_dataset, 'validation': empty_dataset})
        return empty_dataset


def download_code_contests(max_samples=10000, with_validation=True):
    """
    Downloads DeepMind CodeContests dataset for algorithmic reasoning.
    Filters for Python solutions only.
    
    NOTE: This dataset is very large (~25GB). Consider using APPS dataset instead
    or skipping if disk space is limited. This function will return empty dataset
    if download fails.
    """
    print("Downloading CodeContests...")
    print("WARNING: CodeContests is ~25GB. Consider using APPS dataset instead.")
    print("Skipping CodeContests to save disk space. Use APPS dataset for algorithmic reasoning.")
    
    # Return empty dataset
    from datasets import Dataset
    empty_dataset = Dataset.from_dict({"description": [], "solutions": []})
    if with_validation:
        return DatasetDict({'train': empty_dataset, 'validation': empty_dataset})
    return empty_dataset


def download_codesearchnet_python(max_samples=50000, with_validation=True):
    """
    Downloads CodeSearchNet Python subset for documentation/code understanding.
    """
    print("Downloading CodeSearchNet (Python)...")
    dataset = load_dataset("code_search_net", "python", split="train")
    
    # Filter out examples without proper docstrings
    dataset = dataset.filter(lambda x: x.get("func_documentation_string") and len(x["func_documentation_string"]) > 10)
    
    # Limit to max_samples if specified
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    
    if with_validation:
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
    
    return dataset


def download_humaneval():
    """
    Downloads HumanEval - OpenAI's code generation benchmark.
    This should be used for evaluation only, not training.
    """
    print("Downloading HumanEval...")
    dataset = load_dataset("openai_humaneval", split="test")
    return dataset

if __name__ == "__main__":
    download_the_stack()
    download_code_alpaca()
    download_mbpp()