from datasets import load_dataset

def download_the_stack():
    """
    Downloads The Stack dataset.
    """
    print("Downloading The Stack...")
    dataset = load_dataset("bigcode/the-stack", data_dir="data/the-stack", split="train", streaming=True)
    return dataset

def download_code_alpaca(filter_python=False):
    """
    Downloads the Code Alpaca dataset.
    """
    print("Downloading Code Alpaca...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    
    if filter_python:
        # The Code Alpaca dataset doesn't have a language column, but the instructions often mention Python.
        dataset = dataset.filter(lambda example: "python" in example["instruction"].lower())
        
    return dataset

def download_mbpp():
    """
    Downloads the MBPP dataset.
    """
    print("Downloading MBPP...")
    dataset = load_dataset("mbpp", split="test")
    return dataset

if __name__ == "__main__":
    download_the_stack()
    download_code_alpaca()
    download_mbpp()