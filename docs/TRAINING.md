# DyLoRA Training Regimen

This document outlines the current training regimen used in the DyLoRA project.

---

## 1. Current Training Regimen

### Proof of Concept (PoC) Training – [`poc_train.py`](poc_train.py)
- **Model:** `google/gemma-3-270m` with LoRA (`r=8`, `alpha=16`, `dropout=0.1`)
- **Datasets:**
  - Subset of CodeAlpaca (Python-only) with a validation split.
  - MBPP dataset with validation and test splits.
  - Synthetic skill-focused datasets: Requests, Stripe API, Flask.
- **Training Settings:**
  - Continual learning loop over a stream of skill datasets.
  - **Novelty Detection:** Trains a new expert only when a new skill is detected.
  - **Scheduler:** Cosine learning rate scheduler with warmup.
  - **Evaluation:** Evaluates on a validation set at regular intervals (`eval_steps`).
  - **Early Stopping:** Stops training if validation loss does not improve for 5 evaluations.
  - **Checkpointing:** Saves the best model based on validation loss.
  - **Mixed Precision:** Supports `--fp16` and `--bf16`.
- **Cloud Execution:** Via `submit_poc_training.py` on Vertex AI, supports resuming from checkpoint.

### Full Training – [`train.py`](train.py)
- **Model:** `google/codegemma-2b` with LoRA (`r=16`, `alpha=32`, `dropout=0.05`)
- **Datasets:**
  - Full multilingual CodeAlpaca with a validation split.
  - MBPP dataset with validation and test splits.
- **Training Settings:**
  - Standard training loop on the full dataset.
  - **Scheduler:** Cosine learning rate scheduler with warmup.
  - **Evaluation:** Evaluates on the validation set at the end of each epoch.
  - **Early Stopping:** Stops training if validation loss does not improve for 3 epochs.
  - **Checkpointing:** Saves the best model based on validation loss at the end of each epoch.
  - **Mixed Precision:** Supports `--fp16` and `--bf16`.
- **Cloud Execution:** Via [`submit_full_training.py`](submit_full_training.py) on Vertex AI, supports resuming from checkpoint.

---

## 2. Task List for Implementation

- [x] **Dataset refactor** – add validation split for MBPP & CodeAlpaca
- [x] **LoRA configuration tuning** – ranks, alpha, dropout, target modules
- [x] **Mixed precision support** – add `fp16`/`bf16` flags in training
- [x] **Learning rate scheduler** – switch to cosine decay with warmup
- [x] **Validation loop integration** – Hugging Face `Trainer` eval
- [x] **Early stopping hook** – monitor validation loss
- [x] **Checkpoint saving** – every 500 steps + keep best
- [x] **Novelty detection refinement** – remove forced training
- [x] **Vertex AI config** – confirm resumption strategy for spot VM runs
- [x] **Documentation update** – ensure regimen is reflected in README and scripts

---

## 3. Expected Outcomes
- More efficient training with **lower GPU memory usage**.
- Robust training against **spot VM preemptions**.
- Better **generalization** via validation-based tuning.
- Scalable curriculum aligned with novelty-based learning.
- Improved adoption of **LoRA best practices**.
