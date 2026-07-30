import json
import logging
from datetime import datetime

from sqlalchemy import select

from database.model import EmbeddingModel, TransactionModel
from services.agent.sentence_transformer import SentenceTransformerModel
from services.repository.base import get_engine
from shared.config_loader import config_loader

logger = logging.getLogger(__name__)

# The column set every retrieval returns — kept in one place so all tools emit
# the same shape (consumed by _serialize and the agent's auto-analysis step).
_COLUMNS = (
    TransactionModel.transaction_id,
    TransactionModel.amount,
    TransactionModel.event_timestamp,
    TransactionModel.is_fraud,
    TransactionModel.features,
)


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
                "features": r["features"],
            }
            for r in records
        ]
        return json.dumps(payload)

    def _retrieve_context(self, query: str, top_k: int):
        try:
            query_vector = self.embedder.encode(query).tolist()

            stmt = (
                select(
                    TransactionModel.transaction_id,
                    TransactionModel.amount,
                    TransactionModel.event_timestamp,
                    TransactionModel.is_fraud,
                    TransactionModel.features,
                )
                .join(EmbeddingModel, TransactionModel.transaction_id == EmbeddingModel.transaction_id)
                .order_by(EmbeddingModel.embedding.l2_distance(query_vector))
                .limit(top_k)
            )

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Query failed: %s", e)
            raise

    def _retrieve_known_fraud(self, top_k: int):
        try:
            stmt = (
                select(
                    TransactionModel.transaction_id,
                    TransactionModel.amount,
                    TransactionModel.event_timestamp,
                    TransactionModel.is_fraud,
                    TransactionModel.features,
                )
                .where(TransactionModel.is_fraud.is_(True))
                .order_by(TransactionModel.event_timestamp.desc())
                .limit(top_k)
            )

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Known-fraud query failed: %s", e)
            raise

    def _retrieve_filtered(self, conditions: list, top_k: int, query: str | None):
        try:
            stmt = select(*_COLUMNS).where(*conditions)

            if query:
                # Hybrid: filter first, then rank the surviving rows by semantic
                # similarity to the query (only rows that have been embedded).
                query_vector = self.embedder.encode(query).tolist()
                stmt = stmt.join(
                    EmbeddingModel, TransactionModel.transaction_id == EmbeddingModel.transaction_id
                ).order_by(EmbeddingModel.embedding.l2_distance(query_vector))
            else:
                stmt = stmt.order_by(TransactionModel.event_timestamp.desc())

            stmt = stmt.limit(top_k)

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Filtered query failed: %s", e)
            raise

    def query_transactions(
        self,
        top_k: int = 5,
        amount_min: float | None = None,
        amount_max: float | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        is_fraud: bool | None = None,
        query: str | None = None,
    ) -> str:
        """
        Structured (optionally hybrid) transaction lookup. Applies exact SQL
        filters — amount range, time window, fraud flag — and, when ``query`` is
        given, ranks the filtered rows by semantic similarity. This is the
        accurate path for numeric/temporal questions that vector search handles
        poorly.
        """
        top_k = max(1, min(50, top_k))

        conditions = []
        if amount_min is not None:
            conditions.append(TransactionModel.amount >= amount_min)
        if amount_max is not None:
            conditions.append(TransactionModel.amount <= amount_max)
        if is_fraud is not None:
            conditions.append(TransactionModel.is_fraud.is_(is_fraud))

        for label, raw, column, op in (
            ("start_time", start_time, TransactionModel.event_timestamp, "ge"),
            ("end_time", end_time, TransactionModel.event_timestamp, "lt"),
        ):
            if raw is None:
                continue
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                return f"Invalid {label}: '{raw}'. Use ISO-8601, e.g. 2026-07-29T14:00:00."
            conditions.append(column >= parsed if op == "ge" else column < parsed)

        try:
            records = self._retrieve_filtered(conditions, top_k, query)
            result = self._serialize(records)
            logger.info("query_transactions returned %d filter(s)", len(conditions))
            return result
        except Exception as e:
            logger.error("Failed when transforming filtered context: %s", e)
            raise

    def context_lookup(self, query: str, top_k: int = 5) -> str:
        """
        Semantic search for transactions in PostgreSQL by natural-language similarity.
        Will automatically use the top_k specified in the initial request.
        """
        try:
            context = self._retrieve_context(query, top_k)
            result = self._serialize(context)
            logger.info("Retrieved context successfully")
            return result
        except Exception as e:
            logger.error("Failed when transforming context: %s", e)
            raise

    def fraud_lookup(self, top_k: int = 5) -> str:
        """
        Fetch the most recent transactions confirmed as fraudulent (is_fraud = true)
        directly from PostgreSQL, ordered by recency. Ground truth, no vector search.
        """
        try:
            context = self._retrieve_known_fraud(top_k)
            result = self._serialize(context)
            logger.info("Retrieved known-fraud context successfully")
            return result
        except Exception as e:
            logger.error("Failed when transforming known-fraud context: %s", e)
            raise
