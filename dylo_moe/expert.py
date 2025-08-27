from transformers import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

class ExpertManager:
    """
    Manages the creation and application of LoRA experts.
    """
    def __init__(self, model: PreTrainedModel, lora_r: int, lora_alpha: int, lora_dropout: float):
        self.model = model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.num_experts = 0
        self.expert_configs: dict[int, LoraConfig] = {}

    def create_expert(self, r: int | None = None, lora_alpha: int | None = None, lora_dropout: float | None = None) -> int:
        """Create a new LoRA expert with optional overrides.

        Parameters
        ----------
        r: optional rank override
        lora_alpha: optional alpha override
        lora_dropout: optional dropout override
        """
        expert_id = self.num_experts
        
        # Dynamically choose target modules based on model architecture
        common_targets = [
            ["q_proj", "v_proj"],                          # LLaMA/Mistral
            ["query_key_value"],                           # Falcon-style
            ["c_attn"],                                    # GPT-2 style
            ["query", "key", "value"],                     # BERT/RoBERTa
        ]
        chosen_targets = None
        for candidate in common_targets:
            found = False
            try:
                module_names: list[str] = []
                for n, _ in self.model.named_modules():  # type: ignore[attr-defined]
                    try:
                        module_names.append(str(n))  # type: ignore[arg-type]
                    except Exception:
                        continue
                for name in module_names:
                    if any(name.endswith(str(t)) for t in candidate):
                        chosen_targets = candidate
                        found = True
                        break
            except Exception:
                continue
            if found:
                break
        if chosen_targets is None:
            raise ValueError("Could not determine target_modules for LoRA. Please inspect the base model architecture.")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r if r is not None else self.lora_r,
            lora_alpha=lora_alpha if lora_alpha is not None else self.lora_alpha,
            lora_dropout=lora_dropout if lora_dropout is not None else self.lora_dropout,
            target_modules=chosen_targets
        )

        # The peft library directly modifies the model in place
        if not isinstance(self.model, PeftModel):
            # Peft returns a PeftModel subtype; ignore type narrowing complaints
            self.model = get_peft_model(self.model, peft_config, adapter_name=f"expert_{expert_id}")  # type: ignore[assignment]
        else:
            self.model.add_adapter(f"expert_{expert_id}", peft_config)

        self.expert_configs[expert_id] = peft_config
        self.num_experts += 1
        
        return expert_id

    def set_active_expert(self, expert_id: int) -> None:
        """
        Sets the active LoRA expert for the model.
        """
        if isinstance(self.model, PeftModel):
            self.model.set_adapter(f"expert_{expert_id}")