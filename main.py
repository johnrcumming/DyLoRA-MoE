import torch
import os
os.environ["WANDB_DISABLED"] = "false"
import wandb
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from dylo_moe.model import DyLoRA_MoE
from dylo_moe.utils import print_trainable_parameters
from dylo_moe.scheduler import TwoPhaseLRScheduler

def main():
    # 1. Initialize wandb
    wandb.init(project="dylo-moe-software-development")

    # 2. Instantiate the model
    model_name = "gpt2"  # Using a smaller model for testing
    model = DyLoRA_MoE(model_name, num_experts=1)

    # 2. Create a simulated data stream
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the datasets
    from data.prepare_data import download_the_stack, download_code_alpaca
    skill_1_data = download_the_stack()
    skill_2_data = download_code_alpaca()
    
    data_stream = [skill_1_data, skill_2_data]

    # 3. Configure the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        logging_dir='./logs',
        save_safetensors=False,
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

    # 6. Continual learning loop
    for i, skill_data in enumerate(data_stream):
        print(f"\n--- Processing Skill {i+1} ---")
        
        # Create a Dataset object
        tokenized_data = tokenizer(skill_data, padding=True, truncation=True, return_tensors="pt")
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

    # 7. Print the final model architecture and trainable parameters
    print("\n--- Final Model Architecture ---")
    print(model)
    print("\n--- Trainable Parameters ---")
    print_trainable_parameters(model)

if __name__ == "__main__":
    main()