import torch
import os
os.environ["WANDB_DISABLED"] = "false"
import wandb
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from dylo_moe.model import DyLoRA_MoE
from dylo_moe.utils import print_trainable_parameters
from dylo_moe.scheduler import TwoPhaseLRScheduler
from data.prepare_data import download_mbpp

def evaluate(model, tokenizer, dataset):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    total_loss = 0
    device = model.foundation_model.device
    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"]
            loss, outputs = model(**inputs)
            total_loss += loss.item()
    return total_loss / len(dataset)

def main():
    # 1. Initialize wandb
    wandb.init(project="dylo-moe-software-development")

    # 2. Instantiate the model
    model_name = "distilgpt2"  # Using a smaller model for testing
    model = DyLoRA_MoE(model_name, num_experts=1)

    # 2. Create a simulated data stream
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the datasets
    from data.prepare_data import download_code_alpaca
    python_dataset = download_code_alpaca(filter_python=True)
    mbpp_dataset = download_mbpp()
    
    # Create a targeted dataset for the new skill (web scraping with requests)
    requests_dataset = [
        "import requests\n\nresponse = requests.get('https://www.google.com')\nprint(response.status_code)",
        "import requests\n\nresponse = requests.post('https://httpbin.org/post', data = {'key':'value'})\nprint(response.json())"
    ]
    
    python_code = [example["output"] for example in python_dataset.select(range(1000))]
    data_stream = [python_code, requests_dataset]

    # 3. Configure the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        logging_dir='./logs',
        save_safetensors=False,
        gradient_accumulation_steps=4,
    )

    # 4. Instantiate the scheduler
    optimizer = torch.optim.AdamW(
        [
            {"params": model.foundation_model.parameters()},
            {"params": model.router.parameters(), "lr": 1e-3},
        ],
        lr=1e-5
    )
    scheduler = TwoPhaseLRScheduler(
        optimizer,
        high_lr=1e-3,
        low_lr=1e-5,
        seeding_steps=2, # Shortened for this example
        consolidation_steps=3 # Shortened for this example
    )

    # 5. Instantiate the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, scheduler),
    )

    # 6. Initial Evaluation
    print("\n--- Initial Evaluation ---")
    initial_loss = evaluate(model, tokenizer, mbpp_dataset)
    print(f"Initial MBPP Loss: {initial_loss}")
    wandb.log({"initial_mbpp_loss": initial_loss})

    # 7. Continual learning loop
    torch.mps.empty_cache()
    for i, skill_data in enumerate(data_stream):
        print(f"\n--- Processing Skill {i+1} ---")
        
        # Create a Dataset object
        tokenized_data = tokenizer(skill_data, padding=True, truncation=True, return_tensors="pt", max_length=512)
        tokenized_data["labels"] = tokenized_data["input_ids"]
        dataset = Dataset.from_dict(tokenized_data)
        
        # Check for novelty
        device = trainer.model.foundation_model.device
        is_novel = model.add_new_skill(tokenized_data["input_ids"].to(device))
        
        if is_novel:
            # Train on the new skill
            trainer.train_dataset = dataset
            trainer.train()
            
            # Update the expert maturity
            model.router.set_expert_maturity(model.expert_manager.num_experts - 1, 1)

    # 8. Final Evaluation
    print("\n--- Final Evaluation ---")
    final_loss = evaluate(model, tokenizer, mbpp_dataset)
    print(f"Final MBPP Loss: {final_loss}")
    wandb.log({"final_mbpp_loss": final_loss})

    # 9. Print the final model architecture and trainable parameters
    print("\n--- Final Model Architecture ---")
    print(model)
    print("\n--- Trainable Parameters ---")
    print_trainable_parameters(model)

if __name__ == "__main__":
    main()