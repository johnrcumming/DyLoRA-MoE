import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicHybridRouter(nn.Module):
    """
    Implements the Dynamic Hybrid Router for the DyLoRA-MoE architecture.
    """
    def __init__(self, input_size: int, num_experts: int, top_k: int = 1, temperature: float = 2.0):
        super().__init__()
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
        
        # Ensure expert_maturity is on the same device as logits
        if self.expert_maturity.device != logits.device:
            self.expert_maturity = self.expert_maturity.to(logits.device)
        
        # The routing strategy depends on the maturity of the experts
        # If there are new experts or in training mode, use dense collaboration
        # to ensure gradients flow to all experts and the router
        if torch.any(self.expert_maturity == 0) or self.training:
            # Dense collaboration: use a softmax over all experts
            routing_weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            # Sparse delegation (inference only): use a top-k routing
            top_k_weights, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
            
            # Create a sparse tensor with the top-k weights
            routing_weights = torch.zeros_like(logits)
            # scatter maintains gradients through the values being scattered
            routing_weights = routing_weights.scatter(-1, top_k_indices, 
                                                      F.softmax(top_k_weights, dim=-1))

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

    def save(self, path: str):
        """Saves the router's state to a file."""
        state = {
            'gate_state_dict': self.gate.state_dict(),
            'expert_maturity': self.expert_maturity,
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
        router.expert_maturity = state['expert_maturity']
        if device:
            router.to(device)
        return router