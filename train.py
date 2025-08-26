import torch
import os
import argparse
os.environ["WANDB_DISABLED"] = "false"
import wandb
import math
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
from transformers.tokenization_utils_base import BatchEncoding
from typing import Union, Dict, Iterable, Any

from dylo_moe.model import DyLoRA_MoE
from dylo_moe.utils import print_trainable_parameters
from data.prepare_data import download_mbpp

def evaluate(model: torch.nn.Module, tokenizer: AutoTokenizer, dataset: Union[Dataset, DatasetDict]) -> float:
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    total_loss: float = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    examples: Iterable[Dict[str, Any]] = dataset if isinstance(dataset, Dataset) else dataset["train"]
    with torch.no_grad():
        size = 0
        for example in examples:
            if "text" not in example:
                continue
            size += 1
            text = str(example["text"])
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )  # type: ignore
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"]
            outputs = model(**inputs)
            if isinstance(outputs, (tuple, list)):
                loss = outputs[0]
            elif hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                continue
            total_loss += float(loss.item())
    return total_loss / max(1, size)

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

    # 3. Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Load datasets
    from data.prepare_data import download_code_alpaca
    code_alpaca_dataset = download_code_alpaca(filter_python=False, with_validation=True)
    mbpp_dataset = download_mbpp(with_validation=True)

    # 5. Configure the training arguments
    training_args = TrainingArguments(
        output_dir="./results_full",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4,
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
        evaluate_during_training=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
    )

    # 6. Instantiate the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=code_alpaca_dataset["train"],
        eval_dataset=code_alpaca_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 7. Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 8. Final Evaluation on MBPP test set
    print("\n--- Final Evaluation on MBPP ---")
    final_metrics = trainer.evaluate(mbpp_dataset["test"])
    print(f"Final MBPP Test Loss: {final_metrics['eval_loss']}")
    wandb.log({"final_mbpp_test_loss": final_metrics["eval_loss"]})

    # 9. Save the best model
    print("\n--- Saving Best Model ---")
    trainer.save_model("./results_full/best_model")
    tokenizer.save_pretrained("./results_full/best_model")
    print("Best model saved to ./results_full/best_model")

    # 10. Print the final model architecture and trainable parameters
    print("\n--- Final Model Architecture ---")
    print(model)
    print("\n--- Trainable Parameters ---")
    print_trainable_parameters(model)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")
    args = parser.parse_args()
    main(args)