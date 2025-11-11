import os
import torch
import torch.nn as nn
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import DyLoRA_MoE

logger = logging.getLogger(__name__)

def print_trainable_parameters(model: nn.Module):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def log_trainable_parameters(model: nn.Module, prefix: str = ""):
    """
    Logs detailed trainable parameter information for the model.
    
    Args:
        model: PyTorch model to inspect
        prefix: Optional prefix for log messages (e.g., "Before training", "After training")
    """
    try:
        parameters = list(model.parameters())
        num_params = sum(p.numel() for p in parameters)
        num_trainable = sum(p.numel() for p in parameters if p.requires_grad)

        logger.info(
            "%s: Model parameters - Total: %s | Trainable: %s (%.2f%%)",
            prefix,
            f"{num_params:,}",
            f"{num_trainable:,}",
            (num_trainable / num_params * 100) if num_params > 0 else 0.0,
        )
    except Exception as e:
        import warnings
        warnings.warn(f"Could not log model parameters: {e}", RuntimeWarning)

def get_transformer_component(model: nn.Module):
    """
    Get the transformer component from different model architectures.
    
    Args:
        model: Model to extract transformer from
        
    Returns:
        The transformer component (e.g., model.transformer, model.model, etc.)
        
    Raises:
        AttributeError: If no known transformer component is found
    """
    # Common transformer attribute names for different model types
    transformer_attrs = [
        'transformer',      # GPT-2, GPT-Neo, GPT-J
        'model',           # LLaMA, Mistral, Phi
        'gpt_neox',        # GPT-NeoX
        'bert',            # BERT-based models
        'roberta',         # RoBERTa
        'deberta',         # DeBERTa
        'encoder',         # T5 encoder
        'decoder',         # T5 decoder
    ]
    
    for attr in transformer_attrs:
        if hasattr(model, attr):
            return getattr(model, attr)
    
    raise AttributeError(f"Could not find transformer component in {type(model).__name__}")

def save_dylo_moe_state(model: "DyLoRA_MoE", save_directory: str):
    """Saves the router state to a directory."""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save router state
    router_path = os.path.join(save_directory, "router.pt")
    if hasattr(model, 'router') and hasattr(model.router, 'save'):
        model.router.save(router_path)
        print(f"Router state saved to {router_path}")
    else:
        print("Router or router.save method not found. Skipping.")

def save_lora_experts(model: "DyLoRA_MoE", save_directory: str):
    """Saves only the trainable LoRA expert weights and lm_head to a directory."""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for expert_id in range(model.expert_manager.num_experts):
        expert_weights = model.expert_manager.get_expert_weights(expert_id)
        expert_file = os.path.join(save_directory, f"expert_{expert_id}.pt")
        torch.save(expert_weights, expert_file)
    
    print(f"All {model.expert_manager.num_experts} LoRA experts saved to {save_directory}")

def load_lora_experts(model: "DyLoRA_MoE", load_directory: str):
    """Loads LoRA expert weights from a directory into the model."""
    for expert_id in range(model.expert_manager.num_experts):
        expert_file = os.path.join(load_directory, f"expert_{expert_id}.pt")
        if os.path.exists(expert_file):
            expert_weights = torch.load(expert_file, map_location=model.foundation_model.device)
            model.expert_manager.load_expert_weights(expert_id, expert_weights)
            print(f"Loaded weights for expert {expert_id} from {expert_file}")
        else:
            print(f"Warning: No weights file found for expert {expert_id} at {expert_file}")