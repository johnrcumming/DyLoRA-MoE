from datasets import load_dataset

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
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
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

if __name__ == "__main__":
    download_the_stack()
    download_code_alpaca()
    download_mbpp()