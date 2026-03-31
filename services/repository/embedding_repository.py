from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from database.model import TransactionModel, EmbeddingModel
from schemas.transaction import TransactionEmbedding
from services.repository.base import BaseRepository


class TransactionEmbeddingRepository(BaseRepository):
    def fetch_pending(self, batch_size: int) -> list[dict]:
        stmt = (
            select(
                TransactionModel.transaction_id,
                TransactionModel.amount,
                TransactionModel.features,
            )
            .outerjoin(EmbeddingModel, TransactionModel.transaction_id == EmbeddingModel.transaction_id)
            .where(EmbeddingModel.transaction_id == None)
            .limit(batch_size)
        )
        with self.engine.connect() as conn:
            return conn.execute(stmt).mappings().all()

    def save(self, embedding: TransactionEmbedding) -> None:
        data = embedding.model_dump()
        stmt = insert(EmbeddingModel).values(
            transaction_id=data["transaction_id"],
            embedding=data["embedding"],
            embedding_text=data["embedding_text"],
            embedding_model=data["embedding_model"],
        ).on_conflict_do_nothing(index_elements=["transaction_id"])
        with self.engine.begin() as conn:
            conn.execute(stmt)
