import torch

class SkillLibrary:
    """
    Manages a library of skill embeddings for novelty detection.
    """
    def __init__(self, embedding_size: int):
        self.embedding_size = embedding_size
        self.skill_embeddings = {}

    def add_skill(self, skill_id: int, skill_embedding: torch.Tensor):
        """
        Adds a new skill to the library.
        """
        self.skill_embeddings[skill_id] = skill_embedding

    def get_all_skills(self) -> torch.Tensor | None:
        """
        Returns all skill embeddings as a single tensor.
        """
        if not self.skill_embeddings:
            return None
        return torch.stack(list(self.skill_embeddings.values()))