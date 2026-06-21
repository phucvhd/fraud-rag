import os

from services.agent.sentence_transformer import SentenceTransformerModel
from shared.config_loader import config_loader


class EmbeddingProcessor:
    def __init__(self, sentence_transformer_model: SentenceTransformerModel):
        self.cfg = config_loader.load()
        self.model = sentence_transformer_model.get_model()

    def create_embedding(self, amount: float, features: dict, is_fraud: bool) -> tuple[list[float], str]:
        feature_str = ", ".join([f"{k}: {v:.4f}" for k, v in features.items()])
        fraud_status = "CONFIRMED FRAUD" if is_fraud else "normal"
        text_content = f"Transaction of {amount} EUR, fraud status: {fraud_status}. Features: {feature_str}"

        vector = self.model.encode(text_content).tolist()
        return vector, text_content