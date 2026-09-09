import json
import logging

from sqlalchemy import select

from database.model import EmbeddingModel, TransactionModel
from services.agent.sentence_transformer import SentenceTransformerModel
from services.repository.base import get_engine
from shared.config_loader import config_loader

logger = logging.getLogger(__name__)

# Ordering options for find_known_fraud. Kept as a whitelist so the agent (or a
# bad tool call) can never inject an arbitrary column into ORDER BY.
ORDER_RECENCY = "recency"
ORDER_RISK = "risk"
ORDER_AMOUNT = "amount"
_ORDER_BY = {
    ORDER_RECENCY: TransactionModel.event_timestamp.desc(),
    # nulls last: an unscored transaction is not "highest risk".
    ORDER_RISK: TransactionModel.fraud_probability.desc().nullslast(),
    ORDER_AMOUNT: TransactionModel.amount.desc(),
}


def _apply_amount_filters(stmt, amount_min, amount_max):
    """Structured amount filters, applied only when provided so the tools stay
    backward compatible when the agent omits them."""
    if amount_min is not None:
        stmt = stmt.where(TransactionModel.amount >= amount_min)
    if amount_max is not None:
        stmt = stmt.where(TransactionModel.amount <= amount_max)
    return stmt


class RAGQueryEngine:
    def __init__(self, sentence_transformer_model: SentenceTransformerModel):
        self.cfg = config_loader.load()
        self.engine = get_engine(self.cfg.database.url)
        self.embedder = sentence_transformer_model.get_model()

    @staticmethod
    def _serialize(records) -> str:
        if not records:
            return "No data found."

        payload = [
            {
                "transaction_id": str(r["transaction_id"]),
                "event_timestamp": str(r["event_timestamp"]),
                "amount": float(r["amount"]),
                "is_fraud": bool(r["is_fraud"]),
                "fraud_probability": (float(r["fraud_probability"]) if r["fraud_probability"] is not None else None),
                "top_shap_features": r["top_shap_features"],
                "features": r["features"],
                # None on the ground-truth path, which is a SQL filter rather
                # than a vector search — a 0.0 there would read as total
                # retrieval failure on the monitoring dashboard.
                "similarity": (
                    1.0 - float(r["cosine_distance"]) if r.get("cosine_distance") is not None else None
                ),
            }
            for r in records
        ]
        return json.dumps(payload)

    _CONTEXT_COLUMNS = (
        TransactionModel.transaction_id,
        TransactionModel.amount,
        TransactionModel.event_timestamp,
        TransactionModel.is_fraud,
        TransactionModel.fraud_probability,
        TransactionModel.top_shap_features,
        TransactionModel.features,
    )

    def _retrieve_context(self, query=None, top_k: int = 5, amount_min=None, amount_max=None):
        try:
            if query:
                query_vector = self.embedder.encode(query).tolist()
                stmt = (
                    select(
                        *self._CONTEXT_COLUMNS,
                        # Reported for the retriever span only. Ranking still uses
                        # l2 distance so adding observability does not change which
                        # cases the agent sees; cosine is selected because these
                        # embeddings are not L2-normalised, which makes distance
                        # itself uninterpretable as a similarity.
                        EmbeddingModel.embedding.cosine_distance(query_vector).label("cosine_distance"),
                    )
                    .join(EmbeddingModel, TransactionModel.transaction_id == EmbeddingModel.transaction_id)
                )
                # Amount filters narrow BEFORE the similarity limit, so "similar to
                # X AND over 1000 EUR" returns the top_k nearest that also match,
                # not the top_k nearest of which some happen to match.
                stmt = _apply_amount_filters(stmt, amount_min, amount_max)
                stmt = stmt.order_by(EmbeddingModel.embedding.l2_distance(query_vector)).limit(top_k)
            else:
                # No descriptive term — a pure amount-range listing. Skips the
                # vector search entirely (so `similarity` is null) rather than
                # requiring a meaningless query string; ordered by recency.
                stmt = select(*self._CONTEXT_COLUMNS)
                stmt = _apply_amount_filters(stmt, amount_min, amount_max)
                stmt = stmt.order_by(TransactionModel.event_timestamp.desc()).limit(top_k)

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Query failed: %s", e)
            raise

    def _retrieve_known_fraud(self, top_k: int, amount_min=None, amount_max=None, min_risk=None, order_by=ORDER_RECENCY):
        try:
            stmt = select(
                TransactionModel.transaction_id,
                TransactionModel.amount,
                TransactionModel.event_timestamp,
                TransactionModel.is_fraud,
                TransactionModel.fraud_probability,
                TransactionModel.top_shap_features,
                TransactionModel.features,
            ).where(TransactionModel.is_fraud.is_(True))

            stmt = _apply_amount_filters(stmt, amount_min, amount_max)
            if min_risk is not None:
                stmt = stmt.where(TransactionModel.fraud_probability >= min_risk)

            # Unknown order_by falls back to recency rather than erroring — a bad
            # ordering choice should degrade, not fail the investigation.
            order_clause = _ORDER_BY.get(order_by, _ORDER_BY[ORDER_RECENCY])
            stmt = stmt.order_by(order_clause).limit(top_k)

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Known-fraud query failed: %s", e)
            raise

    def _retrieve_suspected(self, top_k: int, min_risk=None, amount_min=None, amount_max=None):
        try:
            # The whole point: filter by the ML score, NOT by is_fraud. This
            # surfaces transactions the model flags as high risk that are not yet
            # confirmed fraud (the chargeback / analyst label has not arrived) —
            # exactly the ones fraud_lookup misses. is_fraud is still returned in
            # the payload so the analyst sees confirmed vs unconfirmed.
            stmt = select(
                TransactionModel.transaction_id,
                TransactionModel.amount,
                TransactionModel.event_timestamp,
                TransactionModel.is_fraud,
                TransactionModel.fraud_probability,
                TransactionModel.top_shap_features,
                TransactionModel.features,
            ).where(TransactionModel.fraud_probability.isnot(None))

            if min_risk is not None:
                stmt = stmt.where(TransactionModel.fraud_probability >= min_risk)
            stmt = _apply_amount_filters(stmt, amount_min, amount_max)
            # Highest model risk first — "most suspicious".
            stmt = stmt.order_by(TransactionModel.fraud_probability.desc()).limit(top_k)

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Suspected-fraud query failed: %s", e)
            raise

    def context_lookup(self, query=None, top_k: int = 5, amount_min=None, amount_max=None) -> str:
        """
        Semantic search for transactions in PostgreSQL by natural-language similarity,
        optionally narrowed to an amount range. When `query` is omitted it becomes a
        pure amount-range listing (no vector search). Uses the top_k from the request.
        """
        try:
            context = self._retrieve_context(query, top_k, amount_min, amount_max)
            result = self._serialize(context)
            logger.info("Retrieved context successfully")
            return result
        except Exception as e:
            logger.error("Failed when transforming context: %s", e)
            raise

    def fraud_lookup(
        self, top_k: int = 5, amount_min=None, amount_max=None, min_risk=None, order_by=ORDER_RECENCY
    ) -> str:
        """
        Fetch transactions confirmed as fraudulent (is_fraud = true) directly from
        PostgreSQL, with optional amount / minimum-risk filters and an ordering
        (recency, risk or amount). Ground truth, no vector search.
        """
        try:
            context = self._retrieve_known_fraud(top_k, amount_min, amount_max, min_risk, order_by)
            result = self._serialize(context)
            logger.info("Retrieved known-fraud context successfully")
            return result
        except Exception as e:
            logger.error("Failed when transforming known-fraud context: %s", e)
            raise

    def suspected_lookup(self, top_k: int = 5, min_risk=None, amount_min=None, amount_max=None) -> str:
        """
        Fetch transactions the ML model scored as high risk, ordered by risk,
        REGARDLESS of the confirmed is_fraud label — including ones not yet
        confirmed as fraud. This is the "catch it before the chargeback" path.
        """
        try:
            context = self._retrieve_suspected(top_k, min_risk, amount_min, amount_max)
            result = self._serialize(context)
            logger.info("Retrieved suspected-fraud context successfully")
            return result
        except Exception as e:
            logger.error("Failed when transforming suspected-fraud context: %s", e)
            raise
