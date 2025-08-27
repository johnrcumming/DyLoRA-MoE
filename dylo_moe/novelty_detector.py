import torch
from .skill_library import SkillLibrary

class NoveltyDetector:
    """Detect skill novelty via cosine similarity thresholding with optional median heuristic."""
    def __init__(
        self,
        skill_library: SkillLibrary,
        similarity_threshold: float = 0.75,
        use_median: bool = False,
    ):
        self.skill_library = skill_library
        self.similarity_threshold = similarity_threshold
        self.use_median = use_median

    def is_novel(self, data_embedding: torch.Tensor) -> bool:
        """Return True if embedding is sufficiently dissimilar from existing skills.

        data_embedding: shape [hidden] (already pooled)
        """
        skill_embeddings = self.skill_library.get_all_skills()
        if skill_embeddings is None:
            return True

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = cos(data_embedding.unsqueeze(0), skill_embeddings)  # [num_skills]

        if self.use_median:
            ref_value = torch.median(similarities)
        else:
            ref_value = torch.max(similarities)

        return bool((ref_value < self.similarity_threshold).item())