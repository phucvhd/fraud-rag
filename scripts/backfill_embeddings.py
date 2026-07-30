import logging

from schemas.transaction import TransactionEmbedding
from services.agent.sentence_transformer import SentenceTransformerModel
from services.embedder.processor import EmbeddingProcessor
from services.repository.embedding_repository import TransactionEmbeddingRepository
from shared.config_loader import config_loader
from shared.logging_config import configure_logging

logger = logging.getLogger("BackfillEmbeddings")


def run():
    cfg = config_loader.load()
    repo = TransactionEmbeddingRepository()
    processor = EmbeddingProcessor(SentenceTransformerModel())

    offset = 0
    total = 0
    while True:
        rows = repo.fetch_all(cfg.database.batch_size, offset)
        if not rows:
            break

        for row in rows:
            vector, text = processor.create_embedding(row["amount"], row["features"], row["is_fraud"])
            repo.upsert(TransactionEmbedding(
                transaction_id=row["transaction_id"],
                embedding=vector,
                embedding_text=text,
                embedding_model=cfg.embedding.model_name,
            ))
            total += 1

        offset += len(rows)
        logger.info("Re-embedded %d transactions so far", total)

    logger.info("Backfill complete: %d transactions re-embedded", total)


if __name__ == "__main__":
    configure_logging()
    run()
