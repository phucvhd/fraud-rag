import json
import logging
from datetime import datetime, timedelta

from sqlalchemy import case, func, select

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


def _apply_time_window(stmt, since_hours, since_days):
    """Restrict to transactions within a recent window, computed server-side.

    Relative (since_hours / since_days) rather than absolute timestamps on
    purpose: the agent maps "last hour" -> since_hours=1 without having to know
    the current time or format a timestamp — a common failure for weak models.
    event_timestamp is a naive column, so the cutoff is naive too.
    """
    if not since_hours and not since_days:
        return stmt
    cutoff = datetime.now() - timedelta(hours=since_hours or 0, days=since_days or 0)
    return stmt.where(TransactionModel.event_timestamp >= cutoff)


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

    def _retrieve_context(self, query=None, top_k: int = 5, amount_min=None, amount_max=None,
                          since_hours=None, since_days=None):
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
                stmt = _apply_time_window(stmt, since_hours, since_days)
                stmt = stmt.order_by(EmbeddingModel.embedding.l2_distance(query_vector)).limit(top_k)
            else:
                # No descriptive term — a pure amount-range listing. Skips the
                # vector search entirely (so `similarity` is null) rather than
                # requiring a meaningless query string; ordered by recency.
                stmt = select(*self._CONTEXT_COLUMNS)
                stmt = _apply_amount_filters(stmt, amount_min, amount_max)
                stmt = _apply_time_window(stmt, since_hours, since_days)
                stmt = stmt.order_by(TransactionModel.event_timestamp.desc()).limit(top_k)

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Query failed: %s", e)
            raise

    def _retrieve_known_fraud(self, top_k: int, amount_min=None, amount_max=None, min_risk=None,
                              order_by=ORDER_RECENCY, since_hours=None, since_days=None):
        try:
            stmt = select(*self._CONTEXT_COLUMNS).where(TransactionModel.is_fraud.is_(True))

            stmt = _apply_amount_filters(stmt, amount_min, amount_max)
            stmt = _apply_time_window(stmt, since_hours, since_days)
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

    def _retrieve_suspected(self, top_k: int, min_risk=None, amount_min=None, amount_max=None,
                            since_hours=None, since_days=None):
        try:
            # The whole point: filter by the ML score, NOT by is_fraud. This
            # surfaces transactions the model flags as high risk that are not yet
            # confirmed fraud (the chargeback / analyst label has not arrived) —
            # exactly the ones fraud_lookup misses. is_fraud is still returned in
            # the payload so the analyst sees confirmed vs unconfirmed.
            stmt = select(*self._CONTEXT_COLUMNS).where(TransactionModel.fraud_probability.isnot(None))

            if min_risk is not None:
                stmt = stmt.where(TransactionModel.fraud_probability >= min_risk)
            stmt = _apply_amount_filters(stmt, amount_min, amount_max)
            stmt = _apply_time_window(stmt, since_hours, since_days)
            # Highest model risk first — "most suspicious".
            stmt = stmt.order_by(TransactionModel.fraud_probability.desc()).limit(top_k)

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Suspected-fraud query failed: %s", e)
            raise

    def _get_by_id(self, transaction_id):
        try:
            stmt = select(*self._CONTEXT_COLUMNS).where(
                TransactionModel.transaction_id == transaction_id
            )
            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Get-by-id query failed: %s", e)
            raise

    def _aggregate_stats(self, since_hours=None, since_days=None, min_risk=0.5):
        try:
            stmt = select(
                func.count().label("total"),
                func.coalesce(func.sum(case((TransactionModel.is_fraud.is_(True), 1), else_=0)), 0).label("fraud"),
                func.coalesce(
                    func.sum(case((TransactionModel.fraud_probability >= min_risk, 1), else_=0)), 0
                ).label("suspected"),
                func.coalesce(func.sum(TransactionModel.amount), 0).label("total_amount"),
            )
            stmt = _apply_time_window(stmt, since_hours, since_days)
            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().one()
        except Exception as e:
            logger.error("Stats query failed: %s", e)
            raise

    def context_lookup(self, query=None, top_k: int = 5, amount_min=None, amount_max=None,
                       since_hours=None, since_days=None) -> str:
        """
        Semantic search for transactions in PostgreSQL by natural-language similarity,
        optionally narrowed to an amount range and/or a recent time window. When
        `query` is omitted it becomes a pure filter listing (no vector search).
        """
        try:
            context = self._retrieve_context(query, top_k, amount_min, amount_max, since_hours, since_days)
            result = self._serialize(context)
            logger.info("Retrieved context successfully")
            return result
        except Exception as e:
            logger.error("Failed when transforming context: %s", e)
            raise

    def get_transaction(self, transaction_id) -> str:
        """Fetch the full record for one specific transaction by its id."""
        try:
            return self._serialize(self._get_by_id(transaction_id))
        except Exception as e:
            logger.error("Failed to fetch transaction %s: %s", transaction_id, e)
            raise

    def fraud_stats(self, since_hours=None, since_days=None, min_risk=0.5) -> str:
        """Aggregate counts over a time window: total transactions, confirmed
        fraud, suspected (score >= min_risk), fraud rate and total amount."""
        try:
            row = self._aggregate_stats(since_hours, since_days, min_risk)
            total = int(row["total"])
            fraud = int(row["fraud"])
            return json.dumps({
                "window": {"since_hours": since_hours, "since_days": since_days},
                "total_transactions": total,
                "confirmed_fraud": fraud,
                "suspected_high_risk": int(row["suspected"]),
                "min_risk_threshold": min_risk,
                # None (not 0) when the window is empty — a rate over zero
                # transactions is undefined, not "0% fraud".
                "fraud_rate": round(fraud / total, 4) if total else None,
                "total_amount": float(row["total_amount"]),
            })
        except Exception as e:
            logger.error("Failed to compute fraud stats: %s", e)
            raise

    def fraud_lookup(
        self, top_k: int = 5, amount_min=None, amount_max=None, min_risk=None, order_by=ORDER_RECENCY,
        since_hours=None, since_days=None
    ) -> str:
        """
        Fetch transactions confirmed as fraudulent (is_fraud = true) directly from
        PostgreSQL, with optional amount / minimum-risk / time-window filters and an
        ordering (recency, risk or amount). Ground truth, no vector search.
        """
        try:
            context = self._retrieve_known_fraud(
                top_k, amount_min, amount_max, min_risk, order_by, since_hours, since_days
            )
            result = self._serialize(context)
            logger.info("Retrieved known-fraud context successfully")
            return result
        except Exception as e:
            logger.error("Failed when transforming known-fraud context: %s", e)
            raise

    def suspected_lookup(self, top_k: int = 5, min_risk=None, amount_min=None, amount_max=None,
                         since_hours=None, since_days=None) -> str:
        """
        Fetch transactions the ML model scored as high risk, ordered by risk,
        REGARDLESS of the confirmed is_fraud label — including ones not yet
        confirmed as fraud. This is the "catch it before the chargeback" path.
        """
        try:
            context = self._retrieve_suspected(top_k, min_risk, amount_min, amount_max, since_hours, since_days)
            result = self._serialize(context)
            logger.info("Retrieved suspected-fraud context successfully")
            return result
        except Exception as e:
            logger.error("Failed when transforming suspected-fraud context: %s", e)
            raise
