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
from datasets import Dataset
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

def preprocess_training_dataset(tokenizer, skill_data):
    """
    Tokenizes the training dataset.
    """
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
        num_train_epochs=args.num_epochs,
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
        max_steps=250,
        remove_unused_columns=False,
    )

    # 5. Instantiate the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        eval_dataset=preprocess_evaluation_dataset(tokenizer, python_dataset["test"]),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # 6. Initial Evaluation
    print("\n--- Initial Evaluation ---")
    tokenized_mbpp_val = preprocess_evaluation_dataset(tokenizer, mbpp_dataset["validation"])
    initial_metrics = trainer.evaluate(tokenized_mbpp_val)
    print(f"Initial MBPP Validation Loss: {initial_metrics['eval_loss']}")
    wandb.log({"initial_mbpp_validation_loss": initial_metrics['eval_loss']})

    # 7. Continual learning loop
    for i, skill_data in enumerate(data_stream):
        print(f"\n--- Processing Skill {i+1} ---")
        
        dataset = preprocess_training_dataset(tokenizer, skill_data)
        
        device = trainer.args.device
        
        is_novel = False
        batch_size = 1
        for j in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[j:j+batch_size]["input_ids"]
            if model.add_new_skill(torch.tensor(batch).to(device)):
                is_novel = True
        
        if is_novel:
            print(f"Novel skill detected. Training new expert...")
            trainer.train_dataset = dataset
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            model.router.set_expert_maturity(model.expert_manager.num_experts - 1, 1)
            wandb.log({"num_experts": model.expert_manager.num_experts})
        else:
            print("Skill not novel. Skipping training.")

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
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()
    main(args)