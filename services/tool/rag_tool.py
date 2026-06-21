import json
import logging

from sqlalchemy import select

from database.model import EmbeddingModel, TransactionModel
from services.agent.sentence_transformer import SentenceTransformerModel
from services.repository.base import get_engine
from shared.config_loader import config_loader

logger = logging.getLogger(__name__)


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
