"""Fit the StandardScaler over the [V1..V28, Amount] feature vector ONCE and
persist it as a JSON artifact (mean/scale). Every transaction is embedded by
subtracting this mean and dividing by this scale, so the scaler must be fit
before the embedder/backfill runs, and re-run only when the corpus changes
enough to warrant it (a re-fit changes the vector space — re-embed afterwards).

Usage: python -m scripts.fit_feature_scaler
"""
import json
import logging
from datetime import datetime, timezone

import numpy as np

from services.embedder.feature_vectorizer import FEATURE_ORDER, VECTOR_DIM, resolve_scaler_path
from services.repository.embedding_repository import TransactionEmbeddingRepository
from shared.config_loader import config_loader
from shared.logging_config import configure_logging

logger = logging.getLogger("FitFeatureScaler")


def run() -> None:
    cfg = config_loader.load()
    repo = TransactionEmbeddingRepository()

    rows_raw: list[np.ndarray] = []
    offset = 0
    while True:
        rows = repo.fetch_all(cfg.database.batch_size, offset)
        if not rows:
            break
        for r in rows:
            row = np.empty(VECTOR_DIM, dtype=np.float64)
            for i, key in enumerate(FEATURE_ORDER):
                row[i] = float(r["amount"]) if key == "Amount" else float(r["features"].get(key, 0.0))
            rows_raw.append(row)
        offset += len(rows)
        logger.info("Loaded %d transactions so far", len(rows_raw))

    if not rows_raw:
        raise SystemExit("No transactions found; populate the table before fitting the scaler.")

    matrix = np.vstack(rows_raw)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    # A constant feature has zero variance; dividing by it yields NaN/inf. Leave
    # such a feature centered (its scaled value is a constant 0 across all rows).
    scale[scale == 0.0] = 1.0

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = {
        "version": version,
        "feature_order": FEATURE_ORDER,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "n_samples": int(matrix.shape[0]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    path = resolve_scaler_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)

    logger.info(
        "Fitted scaler on %d transactions -> %s (version %s). Re-embed to apply.",
        matrix.shape[0], path, version,
    )


if __name__ == "__main__":
    configure_logging()
    run()
