import torch
import os
import argparse
os.environ["WANDB_DISABLED"] = "false"
import wandb
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset, concatenate_datasets
import time
from dylo_moe.model import DyLoRA_MoE
from dylo_moe.utils import print_trainable_parameters
from data.prepare_data import download_mbpp

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
    wandb.init(project="dylo-moe-poc")

    # 2. Instantiate the model
    model_name = "google/gemma-3-270m"
    hf_token = os.environ.get("HF_TOKEN")
    model = DyLoRA_MoE(
        model_name,
        num_experts=1,
        token=hf_token,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    # 3. Create tokenizer and data stream
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    from data.prepare_data import download_code_alpaca, download_mbpp
    python_dataset = download_code_alpaca(filter_python=True, with_validation=True)
    mbpp_dataset = download_mbpp(with_validation=True)
    
    requests_dataset = [
        "import requests\n\nresponse = requests.get('https://www.google.com')\nprint(response.status_code)",
        "import requests\n\nresponse = requests.post('https://httpbin.org/post', data = {'key':'value'})\nprint(response.json())"
    ]
    stripe_dataset = [
        "import stripe\n\nstripe.api_key = 'YOUR_API_KEY'\n\ncharge = stripe.Charge.create(\n  amount=2000,\n  currency='usd',\n  source='tok_mastercard',\n  description='My First Test Charge (created for API docs)',\n)",
        "import stripe\n\nstripe.api_key = 'YOUR_API_KEY'\n\ncustomer = stripe.Customer.create(\n  description='My First Test Customer (created for API docs)',\n)"
    ]
    flask_dataset = [
        "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello_world():\n    return 'Hello, World!'",
        "from flask import Flask, request\n\napp = Flask(__name__)\n\n@app.route('/login', methods=['GET', 'POST'])\ndef login():\n    if request.method == 'POST':\n        return 'POST request'\n    else:\n        return 'GET request'"
    ]
    
    python_code = [example["output"] for example in python_dataset["train"].select(range(1000))]
    data_stream = [python_code, requests_dataset, stripe_dataset, flask_dataset]

    # 4. Configure training arguments
    training_args = TrainingArguments(
        output_dir="./results_poc",
        num_train_epochs=args.num_epochs,  # Reduced from 10
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_dir='./logs_poc',
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=False,
    )

    # 5. Instantiate the trainer
    python_eval = preprocess_evaluation_dataset(tokenizer, python_dataset["test"])
    mbpp_eval = preprocess_evaluation_dataset(tokenizer, mbpp_dataset["validation"])
    combined_eval = concatenate_datasets([
        python_eval.add_column("eval_domain", [0]*len(python_eval)),
        mbpp_eval.add_column("eval_domain", [1]*len(mbpp_eval))
    ])

    # Custom compute_metrics to split domains (0=python,1=mbpp)
    def compute_metrics(eval_pred):
        # eval_pred doesn't contain domain labels, so recompute loss per subset via trainer.evaluate subsets
        # We'll perform manual evaluation below instead; returning empty dict here.
        return {}

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        eval_dataset=combined_eval,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # 6. Initial Evaluation
    print("\n--- Initial Evaluation ---")
    tokenized_mbpp_val = preprocess_evaluation_dataset(tokenizer, mbpp_dataset["validation"])
    initial_metrics = trainer.evaluate(tokenized_mbpp_val)
    print(f"Initial MBPP Validation Loss: {initial_metrics['eval_loss']}")
    wandb.log({"initial_mbpp_validation_loss": initial_metrics['eval_loss']})

    # 7. Continual learning loop
    last_log_time = time.time()
    tokens_processed = 0
    for i, skill_data in enumerate(data_stream):
        print(f"\n--- Processing Skill {i+1} ---")
        
        dataset = preprocess_training_dataset(tokenizer, skill_data, pack=True)
        # Log padding fraction estimate
        pad_tokens = 0
        total_tokens = 0
        for row in dataset["input_ids"]:
            pad_tokens += sum(1 for x in row if x == tokenizer.pad_token_id)
            total_tokens += len(row)
        wandb.log({"padding_fraction": pad_tokens/total_tokens if total_tokens else 0})
        
        device = trainer.args.device
        
        is_novel = False
        batch_size = 1
        for j in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[j:j+batch_size]["input_ids"]
            if model.add_new_skill(torch.tensor(batch).to(device)):
                is_novel = True

        if is_novel:
            print("Novel skill detected. Training new expert...")
            trainer.train_dataset = dataset
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            model.router.set_expert_maturity(model.expert_manager.num_experts - 1, 1)
            wandb.log({"num_experts": model.expert_manager.num_experts})
        else:
            print("Skill not novel. Skipping training.")

        # Log routing metrics if multiple experts
        if model.router.num_experts > 1:
            sample = torch.tensor(dataset[0:1]["input_ids"])
            with torch.no_grad():
                outputs = model.foundation_model(sample, attention_mask=(sample != tokenizer.pad_token_id), output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                routing_weights = model.router(hidden_states)
                entropy = -(routing_weights * (routing_weights.clamp(min=1e-8).log())).sum(-1).mean().item()
                expert_usage = routing_weights.mean(dim=(0,1)).cpu().tolist()
            wandb.log({
                "routing_entropy": entropy,
                **{f"expert_usage_{idx}": val for idx, val in enumerate(expert_usage)}
            })

        # Tokens/sec logging (approx): count non-pad tokens ingested
        for row in dataset["attention_mask"]:
            tokens_processed += sum(row)
        now = time.time()
        if now - last_log_time >= 30:
            tps = tokens_processed / (now - last_log_time)
            wandb.log({"tokens_per_second": tps})
            tokens_processed = 0
            last_log_time = now

        # Evaluate per-domain losses (python vs mbpp) occasionally
        if (i + 1) % 1 == 0:  # every skill
            python_loss = trainer.evaluate(python_eval, metric_key_prefix="eval_python")["eval_python_loss"]
            mbpp_loss = trainer.evaluate(mbpp_eval, metric_key_prefix="eval_mbpp")["eval_mbpp_loss"]
            wandb.log({"eval_python_loss": python_loss, "eval_mbpp_loss": mbpp_loss})

    # 8. Final Evaluation
    print("\n--- Final Evaluation ---")
    tokenized_mbpp_test = preprocess_evaluation_dataset(tokenizer, mbpp_dataset["test"])
    final_metrics = trainer.evaluate(tokenized_mbpp_test)
    print(f"Final MBPP Test Loss: {final_metrics['eval_loss']}")
    wandb.log({"final_mbpp_test_loss": final_metrics['eval_loss']})

    # 9. Save the best model
    print("\n--- Saving Best Model ---")
    trainer.save_model("./results_poc/best_model")
    tokenizer.save_pretrained("./results_poc/best_model")
    print("Best model saved to ./results_poc/best_model")

    # 10. Print the final model architecture and trainable parameters
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()
    main(args)