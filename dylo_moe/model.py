import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftMixedModel
from .router import DynamicHybridRouter
from .expert import ExpertManager
from .novelty_detector import NoveltyDetector
from .skill_library import SkillLibrary

class DyLoRA_MoE(nn.Module):
    """
    Implements the Dynamic LoRA-based Mixture-of-Experts (DyLoRA-MoE) architecture.
    """
    def __init__(self, model_name: str, num_experts: int = 1, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05, token: str | None = None, allow_expert_growth: bool = True, balance_coefficient: float = 0.01):
        super().__init__()
        self.allow_expert_growth = allow_expert_growth
        self.balance_coefficient = balance_coefficient  # Coefficient for load balancing auxiliary loss

        # 1. Load base model
        self.foundation_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            attn_implementation="eager",
        )
        # Disable caching to avoid DynamicCache objects in evaluation padding
        if hasattr(self.foundation_model.config, 'use_cache'):
            self.foundation_model.config.use_cache = False
        
        # Expose config for Trainer compatibility
        self.config = self.foundation_model.config

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

        # 3. Expert manager + initial experts
        self.expert_manager = ExpertManager(self.foundation_model, lora_r, lora_alpha, lora_dropout)
        for _ in range(num_experts):
            self.expert_manager.create_expert()
        
        if num_experts > 0:
            self.expert_manager.set_active_expert(0)

        self.foundation_model = self.expert_manager.model

        # 4. Freeze all non-LoRA params (including lm_head for standard LoRA efficiency)
        for name, param in self.foundation_model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
        self._log_trainable_parameters("After attaching initial experts & freezing base")

        # 5. Router
        hidden_size = int(getattr(self.foundation_model.config, "hidden_size"))  # type: ignore[arg-type]
        self.router = DynamicHybridRouter(input_size=hidden_size, num_experts=num_experts)
        for i in range(num_experts):
            self.router.set_expert_maturity(i, 1)

        # 6. Skill library & novelty detector
        self.skill_library = SkillLibrary(embedding_size=hidden_size)
        self.novelty_detector = NoveltyDetector(self.skill_library)
        
        # 7. Routing tracking
        self.last_routing_weights = None  # For monitoring routing patterns
        self.last_lm_loss = None  # Track language modeling loss separately
        self.last_balance_loss = None  # Track load balancing loss separately

        # 8. Device alignment
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

    def compute_load_balancing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss to encourage uniform expert usage.
        
        This helps prevent routing collapse where one expert dominates and others
        are underutilized. The loss encourages each expert to handle approximately
        1/num_experts of the tokens.
        
        Args:
            routing_weights: Routing weights tensor of shape [batch, seq_len, num_experts]
        
        Returns:
            Scalar tensor representing the load balancing loss (MSE from uniform distribution)
        """
        # Average routing weights across batch and sequence dimensions
        # This gives us the average probability of each expert being selected
        expert_usage = routing_weights.mean(dim=[0, 1])  # [num_experts]
        
        # Target: uniform distribution where each expert gets 1/num_experts of the load
        target = torch.ones_like(expert_usage) / self.router.num_experts
        
        # Compute MSE loss between actual usage and uniform target
        # This penalizes deviation from balanced expert usage
        balance_loss = torch.nn.functional.mse_loss(expert_usage, target)
        
        return balance_loss


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None, expert_id: int | None = None):
        """
        Forward pass of the DyLoRA-MoE model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for language modeling loss
            expert_id: If provided, use this specific expert (for training). 
                      If None, use routing mechanism (for inference).
        """
        # Training mode with explicit expert assignment
        if expert_id is not None:
            self.expert_manager.set_active_expert(expert_id)
            outputs = self.foundation_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            logits = outputs.logits
            
        # Single expert case - direct pass
        elif self.expert_manager.num_experts == 1:
            outputs = self.foundation_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            logits = outputs.logits
            
        # Multi-expert routing with PEFT's single-pass routing_weights
        else:
            # Step 1: Get hidden states from initial forward pass
            # We need hidden states to compute routing weights
            # NOTE: This is 2 forward passes (1 for hidden states, 1 with routing)
            # but still better than the old N+1 passes (1 + N expert loops)
            base_outputs = self.foundation_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = base_outputs.hidden_states[-1]
            
            # Step 2: Compute routing weights from hidden states (with gradients!)
            routing_weights = self.router(hidden_states)  # [batch, seq_len, num_experts]
            
            # Store routing weights for monitoring (detached copy)
            self.last_routing_weights = routing_weights.detach()
            
            # Store non-detached copy for loss computation (to enable gradient flow)
            self._routing_weights_for_loss = routing_weights
            
            # Step 3: Set all experts as active for single-pass routing
            # Use ExpertManager's convenience method
            self.expert_manager.activate_all_experts()
            
            # Step 4: Single forward pass with routing_weights
            # PEFT will automatically combine expert outputs weighted by routing_weights
            outputs = self.foundation_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                routing_weights=routing_weights,  # PEFT's MoE routing
            )
            logits = outputs.logits

        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add load balancing auxiliary loss during training with multi-expert routing
            if self.training and hasattr(self, '_routing_weights_for_loss') and self._routing_weights_for_loss is not None and self.balance_coefficient > 0:
                # Compute load balancing loss to encourage uniform expert usage
                # Use non-detached routing_weights to enable gradient flow to router
                balance_loss = self.compute_load_balancing_loss(self._routing_weights_for_loss)
                
                # Total loss = language modeling loss + weighted load balancing loss
                loss = lm_loss + self.balance_coefficient * balance_loss
                
                # Store individual losses for monitoring (detached to avoid graph issues)
                self.last_lm_loss = lm_loss.detach()
                self.last_balance_loss = balance_loss.detach()
            else:
                # No load balancing during inference or single-expert mode
                loss = lm_loss
                self.last_lm_loss = lm_loss.detach() if lm_loss is not None else None
                self.last_balance_loss = None
            
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # strip caches to prevent accelerator padding issues
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

    from typing import Optional, Dict, Any

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Enables gradient checkpointing for the foundation model.
        """
        self.foundation_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def set_training_mode(self, mode: str = "routing"):
        """
        Set the training mode for the model.
        
        Args:
            mode: One of "routing" (use router to combine experts), 
                  "single_expert" (train one expert at a time via expert_id parameter)
        """
        self.training_mode = mode
        print(f"Training mode set to: {mode}")

    def add_new_skill(self, skill_data: torch.Tensor | None = None, force: bool = False, batch_size: int = 16, warmup_steps: int = 50):
        """
        Adds a new skill to the model.
        If `force` is True, a new expert is created without novelty detection.
        If `allow_expert_growth` is False, no new experts will be created unless forced.
        """
        if not self.allow_expert_growth and not force:
            return False

        is_novel = False
        if force:
            is_novel = True
        elif skill_data is not None:
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
                    norm_layer.to(hidden_states.device)
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
            
            # Add the new skill to the library if we have skill data
            if not force and skill_data is not None:
                self.skill_library.add_skill(new_expert_id, torch.mean(skill_embedding, dim=0))

            # 3. Train the new expert (this will be a separate training loop)
            print(f"New skill detected. Created expert {new_expert_id} (r={dynamic_r}).")
            # Newly added expert remains immature (0) until explicitly matured post-training
            # Ensure mature status of prior experts
            if new_expert_id > 0:
                self.router.set_expert_maturity(new_expert_id - 1, 1)
            
            # Ensure the maturity tensor is on the correct device
            self.router.expert_maturity = self.router.expert_maturity.to(self.foundation_model.device)
        
        return is_novel

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **generate_kwargs):
        """
        Generate text using DyLoRA-MoE with dynamic expert selection.
        
        For generation, we use the router to select the best expert based on the input,
        then generate with that expert active.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            **generate_kwargs: Additional arguments passed to model.generate()
        
        Returns:
            Generated token IDs
        """
        # Set to eval mode for generation
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Single expert case - just use that expert
            if self.expert_manager.num_experts == 1:
                self.expert_manager.set_active_expert(0)
                outputs = self.foundation_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generate_kwargs
                )
            else:
                # Multi-expert: use router to select best expert
                # Get hidden states from a forward pass
                forward_outputs = self.foundation_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = forward_outputs.hidden_states[-1]
                
                # Get routing weights and select the expert with highest average weight
                routing_weights = self.router(hidden_states)  # [batch, seq, num_experts]
                
                # Average routing weights across sequence to get per-batch expert preference
                batch_expert_weights = routing_weights.mean(dim=1)  # [batch, num_experts]
                
                # Select the expert with highest weight for each batch element
                selected_experts = batch_expert_weights.argmax(dim=1)  # [batch]
                
                # For simplicity in generation, use the most common expert across the batch
                # (In a more sophisticated implementation, we could generate separately per batch element)
                expert_id = selected_experts.mode().values.item()
                
                # Activate the selected expert and generate
                self.expert_manager.set_active_expert(expert_id)
                outputs = self.foundation_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generate_kwargs
                )
        
        # Restore training mode
        if was_training:
            self.train()
        
        return outputs