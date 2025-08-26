import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicHybridRouter(nn.Module):
    """
    Implements the Dynamic Hybrid Router for the DyLoRA-MoE architecture.
    """
    def __init__(self, input_size: int, num_experts: int, top_k: int = 1, temperature: float = 2.0):
        super(DynamicHybridRouter, self).__init__()
        self.top_k = top_k
        self.input_size = input_size
        self.num_experts = num_experts
        self.temperature = temperature

        # Gating network
        self.gate = nn.Linear(self.input_size, self.num_experts)

        # State to track expert maturity (0 for new, 1 for mature)
        self.expert_maturity = torch.zeros(self.num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the router.
        """
        # Get the routing weights from the gating network
        logits = self.gate(x)
        
        # The routing strategy depends on the maturity of the experts
        # If there are new experts, use dense collaboration
        # The routing strategy depends on the maturity of the experts
        # If there are new experts, use dense collaboration
        if torch.any(self.expert_maturity == 0):
            # Dense collaboration: use a softmax over all experts
            routing_weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            # Sparse delegation: use a top-k routing
            top_k_weights, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
            
            # Create a sparse tensor with the top-k weights
            routing_weights = torch.zeros_like(logits)
            routing_weights.scatter_(-1, top_k_indices, F.softmax(top_k_weights, dim=-1))

        return routing_weights

    def add_expert(self, device: torch.device | None = None) -> None:
        """
        Adds a new expert to the router.
        """
        self.num_experts += 1
        
        # Resize the gating network
        new_gate = nn.Linear(self.input_size, self.num_experts)
        new_gate.weight.data[:self.num_experts-1, :] = self.gate.weight.data
        new_gate.bias.data[:self.num_experts-1] = self.gate.bias.data
        self.gate = new_gate

        if device:
            self.to(device)

        # Add a new entry to the expert maturity tracker
        self.expert_maturity = torch.cat([self.expert_maturity, torch.zeros(1)])

    def set_expert_maturity(self, expert_id: int, maturity: float) -> None:
        """
        Sets the maturity of an expert.
        """
        self.expert_maturity[expert_id] = maturity