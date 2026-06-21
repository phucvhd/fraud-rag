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
                TransactionModel.is_fraud,
            )
            .outerjoin(EmbeddingModel, TransactionModel.transaction_id == EmbeddingModel.transaction_id)
            .where(EmbeddingModel.transaction_id == None)
            .limit(batch_size)
        )
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings()]

    def fetch_all(self, batch_size: int, offset: int) -> list[dict]:
        stmt = (
            select(
                TransactionModel.transaction_id,
                TransactionModel.amount,
                TransactionModel.features,
                TransactionModel.is_fraud,
            )
            .order_by(TransactionModel.transaction_id)
            .limit(batch_size)
            .offset(offset)
        )
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings()]

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

    def upsert(self, embedding: TransactionEmbedding) -> None:
        data = embedding.model_dump()
        values = {
            "transaction_id": data["transaction_id"],
            "embedding": data["embedding"],
            "embedding_text": data["embedding_text"],
            "embedding_model": data["embedding_model"],
        }
        stmt = insert(EmbeddingModel).values(**values).on_conflict_do_update(
            index_elements=["transaction_id"],
            set_={
                "embedding": values["embedding"],
                "embedding_text": values["embedding_text"],
                "embedding_model": values["embedding_model"],
            },
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
