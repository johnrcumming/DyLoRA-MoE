from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftMixedModel
import torch
from typing import Set, List, Union, Dict

class ExpertManager:
    """
    Manages the creation and application of LoRA experts.
    All LoRA adapters share the same frozen base model weights (handled by PEFT),
    minimizing memory usage. Only the small LoRA adapter weights differ between experts.
    """
    def __init__(self, model: PreTrainedModel, lora_r: int, lora_alpha: int, lora_dropout: float):
        self.model: Union[PreTrainedModel, PeftModel, PeftMixedModel] = model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.num_experts = 0
        self.expert_configs: dict[int, LoraConfig] = {}
        self.current_expert_id: int | None = None

    def create_expert(self, r: int | None = None, lora_alpha: int | None = None, lora_dropout: float | None = None) -> int:
        """
        Create a new LoRA expert. All experts share the same frozen base model weights.
        Only the small LoRA adapter parameters (A and B matrices) are unique per expert.
        """
        expert_id = self.num_experts
        adapter_name = f"expert_{expert_id}"

        # Find target modules dynamically
        target_modules: List[str]
        possible_target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "query_key_value", "c_attn", "Wqkv",
        ]
        
        found_modules: Set[str] = set()
        for name, _ in self.model.named_modules():
            if isinstance(name, str):
                for target in possible_target_modules:
                    if target in name:
                        base_name = name.split('.')[-1]
                        found_modules.add(base_name)

        if found_modules:
            target_modules = list(found_modules)
        else:
            raise ValueError("Could not automatically determine target_modules for LoRA.")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r if r is not None else self.lora_r,
            lora_alpha=lora_alpha if lora_alpha is not None else self.lora_alpha,
            lora_dropout=lora_dropout if lora_dropout is not None else self.lora_dropout,
            target_modules=target_modules
        )

        if not isinstance(self.model, (PeftModel, PeftMixedModel)):
            # First expert: wrap the base model
            self.model = get_peft_model(self.model, peft_config, adapter_name=adapter_name)
        else:
            # Additional experts: add new adapter (shares base weights automatically)
            self.model.add_adapter(adapter_name, peft_config)

        self.expert_configs[expert_id] = peft_config
        self.num_experts += 1
        
        return expert_id

    def set_active_expert(self, expert_id: int) -> None:
        """
        Sets the active LoRA expert by switching which adapter is used.
        All adapters remain in GPU memory and share the same frozen base weights.
        """
        if expert_id == self.current_expert_id:
            return

        if isinstance(self.model, (PeftModel, PeftMixedModel)):
            self.model.set_adapter(f"expert_{expert_id}")
            self.current_expert_id = expert_id

    def get_expert_weights(self, expert_id: int) -> Dict[str, torch.Tensor]:
        """Gets the state dictionary of a specific expert's trainable weights."""
        adapter_name = f"expert_{expert_id}"
        weights: Dict[str, torch.Tensor] = {}
        if not isinstance(self.model, (PeftModel, PeftMixedModel)):
            return weights

        for name, param in self.model.named_parameters():
            if adapter_name in name or "lm_head" in name:
                if param.requires_grad:
                    weights[name] = param.data.clone().cpu()
        return weights

    def load_expert_weights(self, expert_id: int, weights: Dict[str, torch.Tensor]):
        """Loads a state dictionary into a specific expert's weights."""
        adapter_name = f"expert_{expert_id}"
        if not isinstance(self.model, (PeftModel, PeftMixedModel)):
            return

        # Ensure the adapter exists
        if adapter_name not in self.model.peft_config:
            # This assumes the config for the expert was already created
            self.model.add_adapter(adapter_name, self.expert_configs[expert_id])

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.data.copy_(weights[name])