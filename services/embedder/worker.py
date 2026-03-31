import logging
import threading
import time

from schemas.transaction import TransactionEmbedding
from services.agent.sentence_transformer import SentenceTransformerModel
from services.embedder.processor import EmbeddingProcessor
from services.repository.embedding_repository import TransactionEmbeddingRepository
from shared.config_loader import config_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmbeddingWorker")


class EmbeddingWorker:
    def __init__(self, sentence_transformer_model: SentenceTransformerModel):
        self.cfg = config_loader.load()
        self.repo = TransactionEmbeddingRepository()
        self.processor = EmbeddingProcessor(sentence_transformer_model)

    def start(self, stop_event: threading.Event | None = None):
        logger.info("Embedding worker started")
        while not (stop_event and stop_event.is_set()):
            try:
                jobs = self.repo.fetch_pending(self.cfg.database.batch_size)
                if not jobs:
                    time.sleep(2)
                    continue

                for job in jobs:
                    vector, txt = self.processor.create_embedding(
                        job["amount"],
                        job["features"],
                    )
                    self.repo.save(TransactionEmbedding(
                        transaction_id=job["transaction_id"],
                        embedding=vector,
                        embedding_text=txt,
                        embedding_model=self.cfg.embedding.model_name,
                    ))
                    logger.info("Embedded: %s", job["transaction_id"])
            except Exception as e:
                logger.error("Error: %s", e)
                time.sleep(5)
