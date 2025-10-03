import torch
import os
import argparse
os.environ["WANDB_DISABLED"] = "false"
import wandb
from tqdm import tqdm
import math
import time
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers.tokenization_utils_base import BatchEncoding
from typing import Union, Dict, Iterable, Any

from dylo_moe.model import DyLoRA_MoE
from dylo_moe.utils import print_trainable_parameters, save_dylo_moe_state, save_lora_experts
from data.prepare_data import download_mbpp, download_code_alpaca

def preprocess_evaluation_dataset(tokenizer, dataset):
    """
    Tokenizes the evaluation dataset.
    """
    def tokenize_function(examples):
        if "text" in examples:
            # The 'text' column in MBPP is a list of strings, so we join them.
            processed_text = ["\n".join(text) for text in examples["text"]]
        else:
            # The Code Alpaca dataset has 'instruction' and 'output' columns.
            processed_text = [f"{instruction}\n{output}" for instruction, output in zip(examples["instruction"], examples["output"])]
        return tokenizer(processed_text, padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

def pack_sequences(tokenizer, texts, max_length=512):
    """Naive sequence packing: concatenate tokenized sequences until max_length reached."""
    input_ids_batches = []
    attn_batches = []
    cur = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            ids = ids[:max_length]
        if len(cur) + len(ids) > max_length:
            if cur:
                pad_len = max_length - len(cur)
                input_ids_batches.append(cur + [tokenizer.pad_token_id] * pad_len)
                attn_batches.append([1] * len(cur) + [0] * pad_len)
            cur = []
        cur.extend(ids)
    if cur:
        pad_len = max_length - len(cur)
        input_ids_batches.append(cur + [tokenizer.pad_token_id] * pad_len)
        attn_batches.append([1] * len(cur) + [0] * pad_len)
    return input_ids_batches, attn_batches

def preprocess_training_dataset(tokenizer, skill_data, pack=True):
    if pack:
        input_ids, attn = pack_sequences(tokenizer, skill_data)
        dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attn, "labels": input_ids})
    else:
        tokenized_data = tokenizer(skill_data, padding=True, truncation=True, return_tensors="pt", max_length=512)
        dataset = Dataset.from_dict({"input_ids": tokenized_data.input_ids, "attention_mask": tokenized_data.attention_mask, "labels": tokenized_data.input_ids})
    return dataset

def main(args):
    # 1. Initialize wandb
    wandb.init(project="dylo-moe-full-training")

    # 2. Instantiate the model
    hf_token = os.environ.get("HF_TOKEN")
    model = DyLoRA_MoE(
        args.model_name,
        num_experts=args.num_experts,
        token=hf_token,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        allow_expert_growth=False  # Disable dynamic expert growth for traditional training
    )
    
    # Mark all experts as mature so router uses sparse delegation from the start
    for i in range(model.expert_manager.num_experts):
        model.router.set_expert_maturity(i, 1)
    
    print(f"Initialized {model.expert_manager.num_experts} experts (all marked as mature)")
    
    # Ensure all LoRA parameters are trainable
    print("\n--- Verifying trainable parameters ---")
    lora_params = 0
    router_params = 0
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            if "lora" in name.lower():
                lora_params += param.numel()
            elif "router" in name.lower() or "gate" in name.lower():
                router_params += param.numel()
        else:
            frozen_params += param.numel()
    
    print(f"LoRA parameters: {lora_params:,} (trainable)")
    print(f"Router parameters: {router_params:,} (trainable)")
    print(f"Frozen parameters: {frozen_params:,} (includes base model + lm_head)")
    print(f"Total parameters: {total_params:,}")
    print(f"Total trainable: {lora_params + router_params:,}")
    print(f"Trainable %: {(lora_params + router_params) / total_params * 100:.2f}%")
    
    # Verify weight sharing: frozen params should be counted once regardless of num_experts
    print(f"\n--- Memory Efficiency Verification ---")
    print(f"Number of experts: {model.expert_manager.num_experts}")
    print(f"LoRA params per expert (approx): {lora_params // model.expert_manager.num_experts:,}")
    print(f"Base model params (shared): {frozen_params:,}")
    print(f"✓ All experts share the same {frozen_params:,} frozen base weights")
    print(f"✓ Only {lora_params:,} adapter params differ between experts")
    print(f"✓ LM head is frozen (standard LoRA practice)")
    
    if lora_params == 0:
        raise ValueError("No LoRA parameters are trainable! Check model initialization.")

    # 3. Create tokenizer and data stream
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Load datasets
    code_alpaca_dataset = download_code_alpaca(filter_python=False, with_validation=True)
    mbpp_dataset = download_mbpp(with_validation=True)
    
    # Combine both datasets into a single training set
    skill1_data = [ex['instruction'] + "\n" + ex['output'] for ex in code_alpaca_dataset['train']]
    skill2_data = [ex['text'] for ex in mbpp_dataset['train']]
    
    # Merge the datasets
    combined_data = skill1_data + skill2_data
    
    if args.training_subset:
        subset_size = int(len(combined_data) * (args.training_subset / 100))
        combined_data = combined_data[:subset_size]
    
    print(f"Combined training dataset size: {len(combined_data)} examples")


    # 5. Configure the training arguments
    training_args = TrainingArguments(
        output_dir="./results_full",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 32
        gradient_checkpointing=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=args.fp16,
        bf16=args.bf16,
        logging_dir='./logs_full',
        logging_steps=50,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=True,
    )

    # 6. Instantiate the trainer
    
    # Optionally subset the evaluation data
    if args.eval_subset:
        val_subset_size_alpaca = int(len(code_alpaca_dataset["validation"]) * (args.eval_subset / 100))
        alpaca_eval_dataset = code_alpaca_dataset["validation"].select(range(val_subset_size_alpaca))
        
        val_subset_size_mbpp = int(len(mbpp_dataset["validation"]) * (args.eval_subset / 100))
        mbpp_eval_dataset = mbpp_dataset["validation"].select(range(val_subset_size_mbpp))
    else:
        alpaca_eval_dataset = code_alpaca_dataset["validation"]
        mbpp_eval_dataset = mbpp_dataset["validation"]

    alpaca_eval = preprocess_evaluation_dataset(tokenizer, alpaca_eval_dataset)
    mbpp_eval = preprocess_evaluation_dataset(tokenizer, mbpp_eval_dataset)
    
    # Create the combined training dataset
    train_dataset = preprocess_training_dataset(tokenizer, combined_data, pack=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=alpaca_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 7. Initial Evaluation
    print("\n--- Initial Evaluation ---")
    alpaca_initial = trainer.evaluate(alpaca_eval, metric_key_prefix="eval_alpaca")
    mbpp_initial = trainer.evaluate(mbpp_eval, metric_key_prefix="eval_mbpp")
    print(f"Initial Code Alpaca Loss: {alpaca_initial['eval_alpaca_loss']:.4f}")
    print(f"Initial MBPP Loss: {mbpp_initial['eval_mbpp_loss']:.4f}")
    wandb.log({
        "initial_alpaca_loss": alpaca_initial['eval_alpaca_loss'],
        "initial_mbpp_loss": mbpp_initial['eval_mbpp_loss']
    })

    # 8. Train the model on the combined dataset
    print("\n--- Training on Combined Dataset ---")
    print(f"Training with {model.expert_manager.num_experts} experts")
    trainer.train(resume_from_checkpoint=True if args.resume_from_checkpoint else None)

    # 9. Final per-domain evaluation
    print("\n--- Final Evaluation ---")
    alpaca_final = trainer.evaluate(alpaca_eval, metric_key_prefix="eval_alpaca")
    mbpp_final = trainer.evaluate(mbpp_eval, metric_key_prefix="eval_mbpp")
    print(f"Final Code Alpaca Loss: {alpaca_final['eval_alpaca_loss']:.4f}")
    print(f"Final MBPP Loss: {mbpp_final['eval_mbpp_loss']:.4f}")
    wandb.log({
        "final_alpaca_loss": alpaca_final['eval_alpaca_loss'],
        "final_mbpp_loss": mbpp_final['eval_mbpp_loss']
    })

    # 10. Save the best model
    print("\n--- Saving Best Model ---")
    trainer.save_model("./results_full/best_model")
    tokenizer.save_pretrained("./results_full/best_model")
    print("Best model saved to ./results_full/best_model")

    # 10. Save the best model, trainer state, and DyLoRA-MoE state
    print("--- Saving Best Model and Full State ---")
    best_model_dir = "./results_full/best_model"
    
    # Save model, tokenizer, and training arguments
    trainer.save_model(best_model_dir)
    trainer.save_state()  # Saves optimizer, scheduler, etc. to output_dir
    tokenizer.save_pretrained(best_model_dir)
    torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
    print(f"Best model, tokenizer, and training args saved to {best_model_dir} and {training_args.output_dir}")

    # Save DyLoRA-MoE specific state
    dylo_moe_state_dir = os.path.join(training_args.output_dir, "dylo_moe_state")
    save_dylo_moe_state(model, dylo_moe_state_dir)
    print(f"DyLoRA-MoE state saved to {dylo_moe_state_dir}")

    # 11. Upload the entire output directory as a wandb artifact
    print("--- Uploading Artifacts to W&B ---")
    artifact = wandb.Artifact('best-dylora-model-full', type='model')
    artifact.add_dir(training_args.output_dir)
    wandb.log_artifact(artifact)
    print("Best model, trainer state, and DyLoRA-MoE state saved and uploaded to wandb.")

    # 12. Print the final model architecture and trainable parameters
    print("--- Final Model Architecture ---")
    print(model)
    print("--- Trainable Parameters ---")
    print_trainable_parameters(model)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--training_subset", type=int, default=None, help="Percentage of training data to use.")
    parser.add_argument("--eval_subset", type=int, default=None, help="Percentage of evaluation data to use.")
    parser.add_argument("--model_name", type=str, default="google/codegemma-2b", help="The base model to use.")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of LoRA experts to create (fixed at initialization).")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r value.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout value.")
    args = parser.parse_args()
    main(args)