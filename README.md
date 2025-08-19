# DyLoRA-MoE: A Dynamic LoRA-based Mixture-of-Experts Architecture

This repository contains the implementation of the **DyLoRA-MoE**, a Dynamic LoRA-based Mixture-of-Experts architecture designed for continual skill acquisition in large language models.

## Overview

To address the challenge of continual learning and mitigate catastrophic forgetting, the DyLoRA-MoE framework is designed to incrementally acquire new, distinct skills over its lifetime without requiring complete retraining or compromising existing knowledge. It integrates a large, frozen foundation model with a dynamic pool of lightweight, parameter-efficient "experts" based on Low-Rank Adaptation (LoRA).

For more details, please refer to the [Technical Paper](DyLoRA%20-%20Technical%20Paper.md) and the [Technical Design Document](DyLoRA-TDD.md).

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd DyLoRA
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The `main.py` script provides a simple example of how to instantiate and train the DyLoRA-MoE model.

To run the example:
```bash
python main.py
```

This script will:
1.  Instantiate a `DyLoRA_MoE` model with a `gpt2` backbone.
2.  Train the model on a small dummy dataset.
3.  Simulate the acquisition of a new skill.
4.  Print the model architecture and the number of trainable parameters.

## Project Structure

```
.
├── dylo_moe/               # Core Python package for the DyLoRA-MoE implementation
│   ├── __init__.py
│   ├── model.py            # Main DyLoRA_MoE model class
│   ├── router.py           # DynamicHybridRouter implementation
│   ├── expert.py           # ExpertManager for LoRA experts
│   ├── novelty_detector.py # NoveltyDetector implementation
│   ├── scheduler.py        # TwoPhaseLRScheduler implementation
│   └── utils.py            # Utility functions
├── main.py                 # Example script to run the model
├── requirements.txt        # Project dependencies
├── DyLoRA - Technical Paper.md # Project's technical paper
└── DyLoRA-TDD.md           # Technical Design Document