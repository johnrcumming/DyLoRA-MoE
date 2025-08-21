import torch
import os
os.environ["WANDB_DISABLED"] = "false"
import wandb
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
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
            inputs = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"]
            loss, outputs = model(**inputs)
            total_loss += loss.item()
    return total_loss / len(dataset)

def main():
    # 1. Initialize wandb
    wandb.init(project="dylo-moe-software-development-full")

    # 2. Instantiate the model
    model_name = "codellama/CodeLlama-7b-hf"
    model = DyLoRA_MoE(model_name, num_experts=4) # Start with more experts

    # 3. Create a data stream
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load larger, more diverse datasets
    code_alpaca_dataset = load_dataset("the-stack-dedup/code-alpaca-v1", split="train")
    mbpp_dataset = download_mbpp()

    # Create a data stream of different programming languages and tasks
    data_stream = [
        [ex["content"] for ex in code_alpaca_dataset.filter(lambda ex: ex["lang"] == "python").select(range(10000))],
        [ex["content"] for ex in code_alpaca_dataset.filter(lambda ex: ex["lang"] == "javascript").select(range(10000))],
        [ex["content"] for ex in code_alpaca_dataset.filter(lambda ex: ex["lang"] == "java").select(range(10000))],
        [ex["content"] for ex in code_alpaca_dataset.filter(lambda ex: ex["lang"] == "go").select(range(10000))],
    ]

    # 4. Configure the training arguments
    training_args = TrainingArguments(
        output_dir="./results_full",
        num_train_epochs=5, # More epochs for full training
        per_device_train_batch_size=4, # Larger batch size
        logging_dir='./logs_full',
        save_safetensors=True,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
    )

    # 5. Instantiate the optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [
            {"params": model.foundation_model.parameters()},
            {"params": model.router.parameters(), "lr": 5e-4},
        ],
        lr=training_args.learning_rate
    )
    
    # Scheduler will be managed by the Trainer based on TrainingArguments

    # 6. Instantiate the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, None), # Let Trainer handle the scheduler
    )

    # 7. Initial Evaluation
    print("\n--- Initial Evaluation ---")
    initial_loss = evaluate(model, tokenizer, mbpp_dataset)
    print(f"Initial MBPP Loss: {initial_loss}")
    wandb.log({"initial_mbpp_loss": initial_loss})

    # 8. Continual learning loop
    for i, skill_data in enumerate(data_stream):
        print(f"\n--- Processing Skill {i+1} ---")
        
        tokenized_data = tokenizer(skill_data, padding=True, truncation=True, return_tensors="pt", max_length=2048)
        tokenized_data["labels"] = tokenized_data["input_ids"]
        dataset = Dataset.from_dict(tokenized_data)
        
        device = trainer.model.foundation_model.device
        is_novel = model.add_new_skill(tokenized_data["input_ids"].to(device))
        
        if is_novel:
            trainer.train_dataset = dataset
            trainer.train()
            
            model.router.set_expert_maturity(model.expert_manager.num_experts - 1, 1)
            
            wandb.log({
                "train_loss": trainer.state.log_history[-1]["loss"],
                "learning_rate": trainer.state.log_history[-1]["learning_rate"],
                "num_experts": model.expert_manager.num_experts,
            })

    # 9. Final Evaluation
    print("\n--- Final Evaluation ---")
    final_loss = evaluate(model, tokenizer, mbpp_dataset)
    print(f"Final MBPP Loss: {final_loss}")
    wandb.log({"final_mbpp_loss": final_loss})

    # 10. Save and upload the trained LoRA weights
    print("\n--- Saving and Uploading LoRA Weights ---")
    weights_path = "./results_full/dylo_moe_weights.pt"
    torch.save(model.state_dict(), weights_path)
    artifact = wandb.Artifact('dylo-moe-weights', type='model')
    artifact.add_file(weights_path)
    wandb.log_artifact(artifact)
    print("LoRA weights saved and uploaded to wandb.")

    # 11. Print the final model architecture and trainable parameters
    print("\n--- Final Model Architecture ---")
    print(model)
    print("\n--- Trainable Parameters ---")
    print_trainable_parameters(model)

    wandb.finish()

if __name__ == "__main__":
    main()