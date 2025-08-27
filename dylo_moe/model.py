import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .router import DynamicHybridRouter
from .expert import ExpertManager
from .novelty_detector import NoveltyDetector
from .skill_library import SkillLibrary

class DyLoRA_MoE(nn.Module):
    """
    Implements the Dynamic LoRA-based Mixture-of-Experts (DyLoRA-MoE) architecture.
    """
    def __init__(self, model_name: str, num_experts: int = 1, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05, token: str | None = None):
        super(DyLoRA_MoE, self).__init__()

        # 1. Load and freeze the foundation model
        self.foundation_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            attn_implementation="eager"
        )
        
        # Untie the weights of the lm_head
        if self.foundation_model.config.tie_word_embeddings:
            transformer = self._get_transformer()
            # Get the weights from the token embeddings
            # Handle different embedding layer attribute names
            if hasattr(transformer, 'wte'):
                lm_head_weights = transformer.wte.weight
            elif hasattr(transformer, 'embed_tokens'):
                lm_head_weights = transformer.embed_tokens.weight
            else:
                raise AttributeError("Could not find word token embeddings in the transformer.")
            
            # Create a new lm_head with the same dimensions
            self.foundation_model.lm_head = nn.Linear(
                self.foundation_model.config.hidden_size,
                self.foundation_model.config.vocab_size,
                bias=False
            )
            
            # Copy the weights
            self.foundation_model.lm_head.weight = nn.Parameter(lm_head_weights.clone())

        self._log_trainable_parameters("After initializing lm_head")
        
        # 2. Initialize the Expert Manager
        self.expert_manager = ExpertManager(self.foundation_model, lora_r, lora_alpha, lora_dropout)

        # Ensure at least one default expert exists at init
        default_expert_id = self.expert_manager.create_expert()
        self.expert_manager.set_active_expert(default_expert_id)
        
        # Sync foundation_model reference with expert_manager.model (which may now be a PeftModel)
        self.foundation_model = self.expert_manager.model

        # Freeze base model params after LoRA adapters are attached, keep lm_head trainable
        for name, param in self.foundation_model.named_parameters():
            if "lora" not in name.lower() and "lm_head" not in name.lower():
                param.requires_grad = False

        self._log_trainable_parameters("After attaching first LoRA expert and freezing base model")

        # 3. Initialize the Dynamic Hybrid Router
        self.router = DynamicHybridRouter(
            input_size=self.foundation_model.config.hidden_size,
            num_experts=num_experts
        )

        # 4. Initialize the Skill Library
        self.skill_library = SkillLibrary(embedding_size=self.foundation_model.config.hidden_size)

        # 5. Initialize the Novelty Detector
        self.novelty_detector = NoveltyDetector(self.skill_library)
        
        # Move router to the same device as the foundation model
        self.router.to(self.foundation_model.device)
        

    def _log_trainable_parameters(self, prefix: str = ""):
        try:
            parameters = list(self.foundation_model.parameters())
            num_params = sum(p.numel() for p in parameters)
            num_trainable = sum(p.numel() for p in parameters if p.requires_grad)

            # Use logging instead of print for better maintainability
            import logging
            logger = logging.getLogger(__name__)
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

    def _get_transformer(self):
        """
        Get the transformer component from different model architectures.
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
            if hasattr(self.foundation_model, attr):
                return getattr(self.foundation_model, attr)
        
        raise AttributeError(f"Could not find transformer component in {type(self.foundation_model).__name__}")

    def _get_hidden_size(self):
        """
        Get the hidden size from different model configurations.
        """
        config = self.foundation_model.config
        
        # Common hidden size attribute names
        hidden_size_attrs = [
            'hidden_size',      # Most models
            'n_embd',          # GPT-2 style
            'd_model',         # T5 style
            'dim',             # Some custom models
        ]
        
        for attr in hidden_size_attrs:
            if hasattr(config, attr):
                return getattr(config, attr)
        
        raise AttributeError(f"Could not find hidden size in {type(config).__name__}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        """
        Forward pass of the DyLoRA-MoE model.
        """
        # For single expert case, use direct approach for efficiency
        if self.expert_manager.num_experts == 1:
            outputs = self.foundation_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
        else:
            # Multi-expert routing approach
            # Get transformer outputs for routing decisions
            transformer = self._get_transformer()
            transformer_outputs = transformer(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = transformer_outputs.hidden_states[-1]
            
            # Route the input to the appropriate expert(s)
            routing_weights = self.router(hidden_states)
            
            # Apply experts and combine outputs
            expert_logits = []
            for i in range(self.router.num_experts):
                self.expert_manager.set_active_expert(i)
                expert_output = self.foundation_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
                expert_logits.append(expert_output.logits)
            
            # Combine expert outputs using routing weights
            # Average routing weights across sequence length for logit combination
            expert_weights = routing_weights.mean(dim=1)  # [batch_size, num_experts]
            
            logits = torch.zeros_like(expert_logits[0])
            for i, expert_logit in enumerate(expert_logits):
                logits += expert_weights[:, i:i+1, None] * expert_logit
            
            # Use the last expert's outputs for other attributes
            outputs = expert_output
            outputs.logits = logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Create a custom output object
        from transformers.modeling_outputs import CausalLMOutputWithPast
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return (loss, output) if loss is not None else output
    from typing import Optional, Dict, Any

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Enables gradient checkpointing for the foundation model.
        """
        self.foundation_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

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
            
            # Get the embedding for the new skill
            skill_embedding = torch.mean(hidden_states, dim=1)

        is_novel = self.novelty_detector.is_novel(skill_embedding)
        if is_novel:
            # 2. Create a new expert
            new_expert_id = self.expert_manager.create_expert()
            self.router.add_expert(device=self.foundation_model.device)
            self.expert_manager.set_active_expert(new_expert_id)
            
            # Add the new skill to the library
            self.skill_library.add_skill(new_expert_id, torch.mean(skill_embedding, dim=0))

            # 3. Train the new expert (this will be a separate training loop)
            print(f"New skill detected. Created expert {new_expert_id}.")
        
        return is_novel