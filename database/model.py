from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import INT, TIMESTAMP, DECIMAL, BOOLEAN, VARCHAR, ForeignKey, TEXT
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
import datetime
import uuid

class Base(DeclarativeBase):
    pass

class TransactionModel(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_time_seconds: Mapped[int] = mapped_column(INT)
    event_timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP)
    amount: Mapped[float] = mapped_column(DECIMAL(15, 2))
    is_fraud: Mapped[bool] = mapped_column(BOOLEAN, default=False)
    features: Mapped[dict] = mapped_column(JSONB)
    data_source: Mapped[str] = mapped_column(VARCHAR(50))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP, default=datetime.datetime.now())

    embedding: Mapped["EmbeddingModel"] = relationship(back_populates="transaction", uselist=False)

class EmbeddingModel(Base):
    __tablename__ = "transaction_embeddings"

    transaction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("transactions.transaction_id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list] = mapped_column(Vector(384))
    embedding_text: Mapped[str] = mapped_column(TEXT)
    embedding_model: Mapped[str] = mapped_column(VARCHAR(100))
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP, default=datetime.datetime.utcnow)

    transaction: Mapped["TransactionModel"] = relationship(back_populates="embedding")