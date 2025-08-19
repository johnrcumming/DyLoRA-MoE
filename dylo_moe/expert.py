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

    def create_expert(self) -> int:
        """
        Creates a new LoRA expert and adds it to the model.
        """
        expert_id = self.num_experts
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["c_attn"] # This may need to be adjusted based on the foundation model
        )

        # The peft library directly modifies the model in place
        if not isinstance(self.model, PeftModel):
            self.model = get_peft_model(self.model, peft_config, adapter_name=f"expert_{expert_id}")
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