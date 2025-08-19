import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .router import DynamicHybridRouter
from .expert import ExpertManager
from .novelty_detector import NoveltyDetector

class DyLoRA_MoE(nn.Module):
    """
    Implements the Dynamic LoRA-based Mixture-of-Experts (DyLoRA-MoE) architecture.
    """
    def __init__(self, model_name: str, num_experts: int = 1, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super(DyLoRA_MoE, self).__init__()

        # 1. Load and freeze the foundation model
        self.foundation_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Untie the weights of the lm_head
        if self.foundation_model.config.tie_word_embeddings:
            # Get the weights from the token embeddings
            lm_head_weights = self.foundation_model.transformer.wte.weight
            
            # Create a new lm_head with the same dimensions
            self.foundation_model.lm_head = nn.Linear(
                self.foundation_model.config.n_embd,
                self.foundation_model.config.vocab_size,
                bias=False
            )
            
            # Copy the weights
            self.foundation_model.lm_head.weight = lm_head_weights

        for param in self.foundation_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the lm_head for gradient computation
        if hasattr(self.foundation_model, "lm_head"):
            for param in self.foundation_model.lm_head.parameters():
                param.requires_grad = True

        # 2. Initialize the Expert Manager
        self.expert_manager = ExpertManager(self.foundation_model, lora_r, lora_alpha, lora_dropout)

        # 3. Initialize the Dynamic Hybrid Router
        self.router = DynamicHybridRouter(
            input_size=self.foundation_model.config.hidden_size,
            num_experts=num_experts
        )

        # 4. Initialize the Novelty Detector
        self.novelty_detector = NoveltyDetector()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        """
        Forward pass of the DyLoRA-MoE model.
        """
        # Get the transformer outputs without the language modeling head
        transformer_outputs = self.foundation_model.transformer(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]

        # Route the input to the appropriate expert(s)
        routing_weights = self.router(hidden_states)

        # Apply the experts based on the routing weights
        # This is a simplified implementation. A more sophisticated implementation would
        # involve applying the LoRA adapters in a more efficient manner.
        
        # Initialize a tensor to store the final hidden states
        final_hidden_states = torch.zeros_like(hidden_states)

        # Loop through each expert and apply its transformation
        for i in range(self.router.num_experts):
            # Set the active expert
            self.expert_manager.set_active_expert(i)
            
            # Get the output from the foundation model with the active expert
            expert_output = self.foundation_model.transformer(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
            
            # Apply the routing weights
            final_hidden_states += routing_weights[:, :, i].unsqueeze(-1) * expert_output

        # Apply the lm_head to get the final logits
        logits = self.foundation_model.lm_head(final_hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Create a custom output object
        from transformers.modeling_outputs import CausalLMOutputWithPast
        outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=(*transformer_outputs.hidden_states[:-1], final_hidden_states),
            attentions=transformer_outputs.attentions,
        )

        return (loss, outputs) if loss is not None else outputs

    def add_new_skill(self, skill_data: torch.Tensor, batch_size: int = 16):
        """
        Adds a new skill to the model.
        """
        # 1. Detect if the skill is novel
        # Get the router output for the new skill data
        with torch.no_grad():
            all_hidden_states = []
            for i in range(0, skill_data.size(0), batch_size):
                batch = skill_data[i:i+batch_size]
                outputs = self.foundation_model(batch, attention_mask=(batch != 0), output_hidden_states=True)
                all_hidden_states.append(outputs.hidden_states[-1])
            
            hidden_states = torch.cat(all_hidden_states, dim=0)
            router_output = self.router(hidden_states)

        is_novel = self.novelty_detector.is_novel(router_output)
        if is_novel:
            # 2. Create a new expert
            new_expert_id = self.expert_manager.create_expert()
            self.router.add_expert()
            self.expert_manager.set_active_expert(new_expert_id)

            # 3. Train the new expert (this will be a separate training loop)
            print(f"New skill detected. Created expert {new_expert_id}.")
        
        return is_novel