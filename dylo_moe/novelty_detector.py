import torch
from .skill_library import SkillLibrary

class NoveltyDetector:
    """
    Detects if a given input is novel by comparing it to a library of known skills.
    """
    def __init__(self, skill_library: SkillLibrary, similarity_threshold: float = 0.85):
        self.skill_library = skill_library
        self.similarity_threshold = similarity_threshold

    def is_novel(self, data_embedding: torch.Tensor) -> bool:
        """
        Checks if the given data is novel by comparing its embedding to the skill library.
        """
        skill_embeddings = self.skill_library.get_all_skills()

        if skill_embeddings is None:
            return True

        # Calculate cosine similarity
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = cos(data_embedding.unsqueeze(0), skill_embeddings)

        # If the highest similarity is below the threshold, the skill is novel
        return bool((torch.max(similarities) < self.similarity_threshold).item())