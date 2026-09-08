from services.agent.sentence_transformer import SentenceTransformerModel
from shared.config_loader import config_loader


class EmbeddingProcessor:
    def __init__(self, sentence_transformer_model: SentenceTransformerModel):
        self.cfg = config_loader.load()
        self.model = sentence_transformer_model.get_model()

    def _build_text(self, amount: float, features: dict, is_fraud: bool) -> str:
        feature_str = ", ".join([f"{k}: {v:.4f}" for k, v in features.items()])
        fraud_status = "CONFIRMED FRAUD" if is_fraud else "normal"
        return f"Transaction of {amount} EUR, fraud status: {fraud_status}. Features: {feature_str}"

    def create_embedding(self, amount: float, features: dict, is_fraud: bool) -> tuple[list[float], str]:
        text_content = self._build_text(amount, features, is_fraud)
        vector = self.model.encode(text_content).tolist()
        return vector, text_content

    def create_embeddings(self, jobs: list[dict]) -> list[tuple[list[float], str]]:
        """Batch-encode a list of {amount, features, is_fraud} jobs in a single
        model.encode() call — the transformer forward pass is matrix-batched,
        so this is far faster per item than calling create_embedding() in a loop.
        """
        if not jobs:
            return []
        texts = [self._build_text(job["amount"], job["features"], job["is_fraud"]) for job in jobs]
        vectors = self.model.encode(texts)
        return [(vector.tolist(), text) for vector, text in zip(vectors, texts)]
