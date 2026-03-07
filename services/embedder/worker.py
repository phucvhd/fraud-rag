import os
import time
import logging
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import insert

from services.agent.sentence_transformer import SentenceTransformerModel
from services.embedder.processor import EmbeddingProcessor
from shared.config_loader import config_loader
from database.model import TransactionModel, EmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmbeddingWorker")

class EmbeddingWorker:
    def __init__(self, sentence_transformer_model: SentenceTransformerModel):
        self.cfg = config_loader.load()
        self.engine = create_engine(self.cfg.database.url)
        self.processor = EmbeddingProcessor(sentence_transformer_model)

    def _fetch_pending(self):
        stmt = (
            select(
                TransactionModel.transaction_id,
                TransactionModel.amount,
                TransactionModel.features
            )
            .outerjoin(EmbeddingModel, TransactionModel.transaction_id == EmbeddingModel.transaction_id)
            .where(EmbeddingModel.transaction_id == None)
            .limit(self.cfg.database.batch_size)
        )
        with self.engine.connect() as conn:
            return conn.execute(stmt).mappings().all()

    def _save_vector(self, t_id, vector, text_content):
        stmt = insert(EmbeddingModel).values(
            transaction_id=t_id,
            embedding=vector,
            embedding_text=text_content,
            embedding_model=self.cfg.embedding.model_name
        ).on_conflict_do_nothing(index_elements=['transaction_id'])
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def start(self):
        print("Embedding worker started")
        while True:
            try:
                jobs = self._fetch_pending()
                if not jobs:
                    time.sleep(2)
                    continue

                for job in jobs:
                    vector, txt = self.processor.create_embedding(
                        job["amount"],
                        job["features"]
                    )
                    self._save_vector(job["transaction_id"], vector, txt)
                    logger.info(f"Embedded: {job['transaction_id']}")
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(5)
