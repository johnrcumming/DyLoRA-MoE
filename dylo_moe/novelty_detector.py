import torch

class NoveltyDetector:
    """
    Detects if a given input is novel and requires a new expert.
    """
    def __init__(self, entropy_threshold: float = 0.5):
        self.entropy_threshold = entropy_threshold

    def is_novel(self, router_output: torch.Tensor) -> bool:
        """
        Checks if the given data is novel by analyzing the entropy of the router's output.
        A high entropy suggests that the router is uncertain, indicating a new skill.
        """
        # Calculate the entropy of the router's output distribution
        entropy = -torch.sum(router_output * torch.log(router_output + 1e-9), dim=-1)
        
        # Normalize the entropy to be between 0 and 1
        normalized_entropy = entropy / torch.log(torch.tensor(router_output.shape[-1]))
        
        # Check if the average entropy exceeds the threshold
        return bool((torch.mean(normalized_entropy) > self.entropy_threshold).item())