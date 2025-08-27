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
        super().__init__()

        # 1. Load base model
        self.foundation_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            attn_implementation="eager",
        )
        # Disable caching to avoid DynamicCache objects in evaluation padding
        if hasattr(self.foundation_model.config, 'use_cache'):
            self.foundation_model.config.use_cache = False

        # 2. Untie lm_head if tied
        if getattr(self.foundation_model.config, "tie_word_embeddings", False):
            transformer = self._get_transformer()
            if hasattr(transformer, "wte"):
                lm_head_weights = transformer.wte.weight
            elif hasattr(transformer, "embed_tokens"):
                lm_head_weights = transformer.embed_tokens.weight
            else:
                raise AttributeError("Could not find word token embeddings in transformer for untie.")
            self.foundation_model.lm_head = nn.Linear(
                self.foundation_model.config.hidden_size,
                self.foundation_model.config.vocab_size,
                bias=False,
            )
            self.foundation_model.lm_head.weight = nn.Parameter(lm_head_weights.clone())
        self._log_trainable_parameters("After initializing lm_head")

        # 3. Expert manager + first expert
        self.expert_manager = ExpertManager(self.foundation_model, lora_r, lora_alpha, lora_dropout)
        first_expert_id = self.expert_manager.create_expert()
        self.expert_manager.set_active_expert(first_expert_id)
        self.foundation_model = self.expert_manager.model

        # 4. Freeze non-LoRA params except lm_head
        for name, param in self.foundation_model.named_parameters():
            if "lora" not in name.lower() and "lm_head" not in name.lower():
                param.requires_grad = False
        self._log_trainable_parameters("After attaching first expert & freezing base")

        # 5. Router
        hidden_size = int(getattr(self.foundation_model.config, "hidden_size"))  # type: ignore[arg-type]
        self.router = DynamicHybridRouter(input_size=hidden_size, num_experts=num_experts)
        if self.router.expert_maturity.numel() > 0:
            self.router.expert_maturity[0] = 1

        # 6. Skill library & novelty detector
        self.skill_library = SkillLibrary(embedding_size=hidden_size)
        self.novelty_detector = NoveltyDetector(self.skill_library)

        # 7. Device alignment
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
                use_cache=False,
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
                    use_cache=False,
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
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # strip caches to prevent accelerator padding issues
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    from typing import Optional, Dict, Any

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Enables gradient checkpointing for the foundation model.
        """
        self.foundation_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def add_new_skill(self, skill_data: torch.Tensor, batch_size: int = 16, warmup_steps: int = 50):
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
            _ = self.router(hidden_states)  # force same path, discard
            # Improved embedding: token mean after RMSNorm (if available) else LayerNorm fallback
            if hasattr(self.foundation_model, 'model') and hasattr(self.foundation_model.model, 'norm'):
                norm_layer = self.foundation_model.model.norm  # type: ignore[attr-defined]
                normed = norm_layer(hidden_states)
            else:
                norm_layer = torch.nn.LayerNorm(hidden_states.size(-1), eps=1e-6)
                normed = norm_layer(hidden_states)
            skill_embedding = normed.mean(dim=1)  # [batch, hidden]

        is_novel = self.novelty_detector.is_novel(skill_embedding)
        if is_novel:
            # 2. Create a new expert
            # Increase adapter capacity slightly for later experts
            dynamic_r = self.expert_manager.lora_r if self.expert_manager.num_experts == 0 else min(self.expert_manager.lora_r * 2, 32)
            new_expert_id = self.expert_manager.create_expert(r=dynamic_r)
            self.router.add_expert(device=self.foundation_model.device)
            self.expert_manager.set_active_expert(new_expert_id)
            
            # Add the new skill to the library
            self.skill_library.add_skill(new_expert_id, torch.mean(skill_embedding, dim=0))

            # 3. Train the new expert (this will be a separate training loop)
            print(f"New skill detected. Created expert {new_expert_id} (r={dynamic_r}).")
            # Newly added expert remains immature (0) until explicitly matured post-training
            # Ensure mature status of prior experts
            if new_expert_id > 0:
                self.router.set_expert_maturity(new_expert_id - 1, 1)
        
        return is_novel