from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from uuid import UUID, uuid4
from datetime import datetime

class TransactionBase(BaseModel):
    transaction_id: UUID = Field(default_factory=uuid4)
    event_time_seconds: int
    amount: float = Field(..., gt=0)
    features: Dict[str, float]
    is_fraud: bool = False
    data_source: str

class TransactionCanonical(TransactionBase):
    event_timestamp: datetime
    created_at: datetime = datetime.now()

    @field_validator('event_timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

class TransactionEmbedding(BaseModel):
    transaction_id: UUID
    embedding: Optional[List[float]] = None
    embedding_text: str
    embedding_model: str
    created_at: datetime = datetime.now()