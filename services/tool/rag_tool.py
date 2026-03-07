import logging

from sqlalchemy import create_engine, select

from services.agent.sentence_transformer import SentenceTransformerModel
from shared.config_loader import config_loader
from database.model import TransactionModel, EmbeddingModel

logger = logging.getLogger(__name__)

class RAGQueryEngine:
    def __init__(self, sentence_transformer_model: SentenceTransformerModel):
        self.cfg = config_loader.load()
        self.engine = create_engine(self.cfg.database.url)
        self.embedder = sentence_transformer_model.get_model()

    def _retrieve_context(self, query: str, top_k: int):
        try:
            query_vector = self.embedder.encode(query).tolist()

            stmt = (
                select(
                    TransactionModel.transaction_id,
                    TransactionModel.amount,
                    TransactionModel.event_timestamp,
                    EmbeddingModel.embedding_text
                )
                .join(EmbeddingModel, TransactionModel.transaction_id == EmbeddingModel.transaction_id)
                .order_by(EmbeddingModel.embedding.l2_distance(query_vector))
                .limit(top_k)
            )

            with self.engine.connect() as conn:
                return conn.execute(stmt).mappings().all()
        except Exception as e:
            logger.error("Query has been failed", e)
            raise e

    def context_lookup(self, query: str, top_k: int = 5) -> str:
        """
        Search for past fraud cases in PostgreSQL.
        Will automatically use the top_k specified in the initial request.
        """
        try:
            context = self._retrieve_context(query, top_k)
            if not context:
                return "No data found."

            context_text = "\n".join([
                f"TransactionId: {r['transaction_id']} | Time: {r['event_timestamp']} | Amount: {r['amount']} | Details: {r['embedding_text']}"
                for r in context
            ])
            logger.info("Retrieve context successfully")
            return context_text
        except Exception as e:
            logger.error("Failed when transforming context", e)
            raise e
