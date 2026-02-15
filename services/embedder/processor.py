import os

from sentence_transformers import SentenceTransformer
from shared.config_loader import config_loader


class EmbeddingProcessor:
    def __init__(self):
        self.cfg = config_loader.load()
        cache_dir = os.getenv("HF_HOME", None)

        self.model = SentenceTransformer(
            self.cfg.embedding.model_name,
            cache_folder=cache_dir
        )

    def create_embedding(self, amount: float, features: dict) -> tuple[list[float], str]:
        feature_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(features.items())[:5]])
        text_content = f"Transaction of {amount} EUR. Key features: {feature_str}"

        vector = self.model.encode(text_content).tolist()
        return vector, text_content