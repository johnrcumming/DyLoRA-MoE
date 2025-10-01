import os
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import DyLoRA_MoE

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

def save_dylo_moe_state(model: "DyLoRA_MoE", save_directory: str):
    """Saves the router and skill library state to a directory."""
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save router state
    router_path = os.path.join(save_directory, "router.pt")
    if hasattr(model, 'router') and hasattr(model.router, 'save'):
        model.router.save(router_path)
        print(f"Router state saved to {router_path}")
    else:
        print("Router or router.save method not found. Skipping.")

    # Save skill library
    skill_library_path = os.path.join(save_directory, "skill_library.pt")
    if hasattr(model, 'skill_library') and hasattr(model.skill_library, 'save'):
        model.skill_library.save(skill_library_path)
        print(f"Skill library saved to {skill_library_path}")
    else:
        print("Skill library or skill_library.save method not found. Skipping.")

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