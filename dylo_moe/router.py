import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicHybridRouter(nn.Module):
    """
    Implements the Dynamic Hybrid Router for the DyLoRA-MoE architecture.
    
    The router learns to assign weights to experts based on input hidden states.
    During training, uses dense softmax routing to ensure all experts receive gradients.
    During inference, can use sparse top-k routing for efficiency.
    """
    def __init__(self, input_size: int, num_experts: int, top_k: int = 1, temperature: float = 2.0):
        super().__init__()
        self.top_k = top_k
        self.input_size = input_size
        self.num_experts = num_experts
        self.temperature = temperature

        # Gating network
        self.gate = nn.Linear(self.input_size, self.num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the router.
        
        Routes inputs to experts using dense (training) or sparse (inference) strategy:
        - Training: Dense softmax over all experts to ensure gradient flow
        - Inference: Sparse top-k selection for computational efficiency
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size]
            
        Returns:
            routing_weights: Tensor of shape [batch, seq_len, num_experts] with expert weights
        """
        # Get the routing weights from the gating network
        logits = self.gate(x)
        
        # Use training flag to determine routing strategy:
        # - Dense during training for gradient flow to all experts and router
        # - Sparse during inference for efficiency
        if self.training:
            # Dense collaboration: use a softmax over all experts
            routing_weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            # Sparse delegation (inference only): use a top-k routing
            top_k_weights, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
            
            # Create a sparse tensor with the top-k weights
            # Ensure dtype matches for scatter operation (important for mixed precision)
            routing_weights = torch.zeros_like(logits)
            # scatter maintains gradients through the values being scattered
            # Explicitly match dtype for scatter operation (fixes mixed precision issues)
            softmax_weights = F.softmax(top_k_weights, dim=-1).to(routing_weights.dtype)
            routing_weights = routing_weights.scatter(-1, top_k_indices, softmax_weights)

        return routing_weights

    def save(self, path: str):
        """Saves the router's state to a file."""
        state = {
            'gate_state_dict': self.gate.state_dict(),
            'num_experts': self.num_experts,
            'input_size': self.input_size,
            'top_k': self.top_k,
            'temperature': self.temperature
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: torch.device | None = None):
        """Loads a router's state from a file."""
        state = torch.load(path, map_location=device)
        router = cls(
            input_size=state['input_size'],
            num_experts=state['num_experts'],
            top_k=state['top_k'],
            temperature=state['temperature']
        )
        router.gate.load_state_dict(state['gate_state_dict'])
        if device:
            router.to(device)
        return router