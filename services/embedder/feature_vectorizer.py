import json
import logging
from pathlib import Path

import numpy as np

from shared.config_loader import config_loader

logger = logging.getLogger(__name__)

# Project root: services/embedder/feature_vectorizer.py -> parents[2].
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical vector layout. This order is FROZEN: the scaler artifact, every
# stored vector and every query must agree on it, so it lives in exactly one
# place. V1..V28 are the dataset's PCA components; Amount is appended so the
# transaction value is part of the geometry rather than a separate column.
FEATURE_ORDER: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
VECTOR_DIM: int = len(FEATURE_ORDER)  # 29


def resolve_scaler_path(path: str | None = None) -> Path:
    """Resolve the scaler artifact path against the project root when relative,
    so it does not depend on the process working directory."""
    p = Path(path or config_loader.load().embedding.scaler_path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


class FeatureVectorizer:
    """Turns a transaction's raw PCA features + amount into the standardized
    numeric vector stored in pgvector.

    The vector is NOT a text embedding: it is ``[V1..V28, Amount]`` scaled by a
    StandardScaler that was fit ONCE over the whole corpus (mean/scale persisted
    in a JSON artifact). Fitting once — rather than per batch — is what makes the
    distance between any two vectors comparable: every vector lives on the same
    axes with the same units. Because it is a real feature-space representation,
    L2 distance between two vectors reflects how similar the transactions are in
    behaviour, which a text embedding of stringified numbers never guaranteed.
    """

    def __init__(self, scaler_path: str | None = None):
        self.path = resolve_scaler_path(scaler_path)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.version: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Feature scaler artifact not found at {self.path}. Run "
                "`python -m scripts.fit_feature_scaler` after the transactions "
                "table is populated to fit and persist it."
            )
        with open(self.path) as f:
            art = json.load(f)
        if art.get("feature_order") != FEATURE_ORDER:
            raise ValueError(
                "Scaler artifact feature_order does not match FEATURE_ORDER; "
                "refit the scaler with scripts/fit_feature_scaler.py."
            )
        self._mean = np.asarray(art["mean"], dtype=np.float64)
        self._scale = np.asarray(art["scale"], dtype=np.float64)
        self.version = art["version"]

    @property
    def model_descriptor(self) -> str:
        """Provenance string stored in transaction_embeddings.embedding_model —
        records WHICH scaler produced a given vector, so a re-fit is detectable."""
        return f"standardscaler-{self.version}"

    @staticmethod
    def raw_row(amount: float, features: dict) -> np.ndarray:
        """Ordered, unscaled 29-vector. A feature absent from the payload falls
        back to 0.0 (its post-scaling mean is ~0 anyway); extra keys are ignored,
        so upstream schema drift degrades instead of crashing."""
        row = np.empty(VECTOR_DIM, dtype=np.float64)
        for i, key in enumerate(FEATURE_ORDER):
            row[i] = float(amount) if key == "Amount" else float(features.get(key, 0.0))
        return row

    def transform(self, amount: float, features: dict) -> list[float]:
        row = self.raw_row(amount, features)
        return ((row - self._mean) / self._scale).tolist()

    def transform_many(self, jobs: list[dict]) -> list[list[float]]:
        if not jobs:
            return []
        raw = np.vstack([self.raw_row(j["amount"], j["features"]) for j in jobs])
        return ((raw - self._mean) / self._scale).tolist()
