import os

from services.agent.sentence_transformer import SentenceTransformerModel
from shared.config_loader import config_loader


class EmbeddingProcessor:
    def __init__(self, sentence_transformer_model: SentenceTransformerModel):
        self.cfg = config_loader.load()
        self.model = sentence_transformer_model.get_model()

    def create_embedding(self, amount: float, features: dict) -> tuple[list[float], str]:
        feature_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(features.items())[:5]])
        text_content = f"Transaction of {amount} EUR. Key features: {feature_str}"

        vector = self.model.encode(text_content).tolist()
        return vector, text_content