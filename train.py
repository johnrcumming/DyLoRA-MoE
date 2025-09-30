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
from dylo_moe.utils import print_trainable_parameters
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
    model_name = "google/codegemma-2b"
    hf_token = os.environ.get("HF_TOKEN")
    model = DyLoRA_MoE(
        model_name,
        num_experts=1,
        token=hf_token,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )

    # 3. Create tokenizer and data stream
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Load datasets
    code_alpaca_dataset = download_code_alpaca(filter_python=False, with_validation=True)
    mbpp_dataset = download_mbpp(with_validation=True)
    
    # Create a data stream similar to poc_train.py
    skill1_data = [ex['instruction'] + "\n" + ex['output'] for ex in code_alpaca_dataset['train']]
    skill2_data = [ex['text'] for ex in mbpp_dataset['train']]
    data_stream = [skill1_data, skill2_data]


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
    alpaca_eval = preprocess_evaluation_dataset(tokenizer, code_alpaca_dataset["validation"])
    mbpp_eval = preprocess_evaluation_dataset(tokenizer, mbpp_dataset["validation"])
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_dict({}), # Dummy dataset, will be replaced in the loop
        eval_dataset=alpaca_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 7. Initial Evaluation
    print("\n--- Initial Evaluation ---")
    initial_metrics = trainer.evaluate(mbpp_eval)
    print(f"Initial MBPP Validation Loss: {initial_metrics['eval_loss']}")
    wandb.log({"initial_mbpp_validation_loss": initial_metrics['eval_loss']})

    # 8. Continual learning loop
    for i, skill_data in enumerate(data_stream):
        print(f"\n--- Processing Skill {i+1} ---")
        
        dataset = preprocess_training_dataset(tokenizer, skill_data, pack=True)
        
        device = trainer.args.device
        
        # We are forcing a new expert for each skill in the data stream
        is_novel = model.add_new_skill(force=True)

        if is_novel:
            print("Novel skill detected. Training new expert...")
            trainer.train_dataset = dataset
            trainer.eval_dataset = alpaca_eval if i == 0 else mbpp_eval # Evaluate on the corresponding dataset
            trainer.train(resume_from_checkpoint=True if args.resume_from_checkpoint else None)
            model.router.set_expert_maturity(model.expert_manager.num_experts - 1, 1)
            wandb.log({"num_experts": model.expert_manager.num_experts})
        else:
            print("Skill not novel. Skipping training.")

        # Evaluate per-domain losses
        alpaca_loss = trainer.evaluate(alpaca_eval, metric_key_prefix="eval_alpaca")["eval_alpaca_loss"]
        mbpp_loss = trainer.evaluate(mbpp_eval, metric_key_prefix="eval_mbpp")["eval_mbpp_loss"]
        wandb.log({"eval_alpaca_loss": alpaca_loss, "eval_mbpp_loss": mbpp_loss})


    # 9. Final Evaluation on MBPP test set
    print("\n--- Final Evaluation on MBPP ---")
    mbpp_test_eval = preprocess_evaluation_dataset(tokenizer, mbpp_dataset["test"])
    final_metrics = trainer.evaluate(mbpp_test_eval)
    print(f"Final MBPP Test Loss: {final_metrics['eval_loss']}")
    wandb.log({"final_mbpp_test_loss": final_metrics["eval_loss"]})

    # 10. Save the best model
    print("\n--- Saving Best Model ---")
    trainer.save_model("./results_full/best_model")
    tokenizer.save_pretrained("./results_full/best_model")
    print("Best model saved to ./results_full/best_model")

    # 11. Print the final model architecture and trainable parameters
    print("\n--- Final Model Architecture ---")
    print(model)
    print("\n--- Trainable Parameters ---")
    print_trainable_parameters(model)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint.")
    args = parser.parse_args()
    main(args)