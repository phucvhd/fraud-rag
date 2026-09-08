from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    prompt: str
    top_k: int = Field(default=5, gt=0, le=50)


class QueryResponse(BaseModel):
    answer: str


class TimeseriesBucket(BaseModel):
    bucket: str
    transactions: int
    fraud: int
    normal: int


class TimeseriesResponse(BaseModel):
    data: list[TimeseriesBucket]
    total_transactions: int
    total_fraud: int
    total_normal: int


class TransactionRecord(BaseModel):
    transaction_id: UUID
    event_timestamp: datetime
    amount: float
    is_fraud: bool
    fraud_probability: float | None = None
    data_source: str


class TransactionListResponse(BaseModel):
    data: list[TransactionRecord]
    total: int