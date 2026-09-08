import logging
import threading
import time

from schemas.transaction import TransactionEmbedding
from services.agent.sentence_transformer import SentenceTransformerModel
from services.embedder.processor import EmbeddingProcessor
from services.repository.embedding_repository import TransactionEmbeddingRepository
from services.repository.status_repository import TransactionStatusRepository
from shared.config_loader import config_loader
from shared.logging_config import configure_logging

logger = logging.getLogger("EmbeddingWorker")


class EmbeddingWorker:
    def __init__(self, sentence_transformer_model: SentenceTransformerModel):
        self.cfg = config_loader.load()
        self.repo = TransactionEmbeddingRepository()
        self.status_repo = TransactionStatusRepository()
        self.processor = EmbeddingProcessor(sentence_transformer_model)

    def start(self, stop_event: threading.Event | None = None):
        logger.info("Embedding worker started")
        while not (stop_event and stop_event.is_set()):
            try:
                jobs = self.repo.fetch_pending(self.cfg.database.batch_size)
                if not jobs:
                    time.sleep(2)
                    continue

                job_ids = [str(job["transaction_id"]) for job in jobs]
                self.status_repo.mark_embedding(job_ids)

                embeddings = self.processor.create_embeddings(jobs)
                records = [
                    TransactionEmbedding(
                        transaction_id=job["transaction_id"],
                        embedding=vector,
                        embedding_text=txt,
                        embedding_model=self.cfg.embedding.model_name,
                    )
                    for job, (vector, txt) in zip(jobs, embeddings)
                ]
                self.repo.save_many(records)
                self.status_repo.mark_embedded(job_ids)
                logger.info("Embedded %d transactions", len(records))
            except Exception as e:
                logger.error("Error: %s", e)
                time.sleep(5)


if __name__ == "__main__":
    configure_logging()
    worker = EmbeddingWorker(SentenceTransformerModel())
    worker.start()
