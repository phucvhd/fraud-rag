from services.embedder.feature_vectorizer import FEATURE_ORDER, FeatureVectorizer


class EmbeddingProcessor:
    def __init__(self, feature_vectorizer: FeatureVectorizer):
        self.vectorizer = feature_vectorizer

    @property
    def model_descriptor(self) -> str:
        return self.vectorizer.model_descriptor

    def _provenance_text(self, amount: float, features: dict) -> str:
        """Human-readable record of WHAT was vectorized, stored in embedding_text
        for debugging. Deliberately does NOT include is_fraud: the label must
        never enter the representation (that leaks the answer into retrieval).
        This text is provenance only — the standardized numeric vector, not this
        string, is what defines similarity."""
        feature_str = ", ".join(
            f"{k}: {float(features.get(k, 0.0)):.4f}" for k in FEATURE_ORDER if k != "Amount"
        )
        return f"Transaction of {amount} EUR. Features: {feature_str}"

    def create_embedding(self, amount: float, features: dict) -> tuple[list[float], str]:
        vector = self.vectorizer.transform(amount, features)
        return vector, self._provenance_text(amount, features)

    def create_embeddings(self, jobs: list[dict]) -> list[tuple[list[float], str]]:
        """Batch-scale a list of {amount, features} jobs in one matrix operation,
        far faster per item than transforming one at a time."""
        if not jobs:
            return []
        vectors = self.vectorizer.transform_many(jobs)
        return [
            (vector, self._provenance_text(job["amount"], job["features"]))
            for vector, job in zip(vectors, jobs)
        ]
