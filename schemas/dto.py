from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    prompt: str
    top_k: int = Field(default=5, gt=0, le=50)
    # Passed through to the trace so production traffic can be grouped per
    # conversation and per analyst — without them, per-user cost and error
    # rates are not answerable.
    session_id: str | None = Field(default=None, max_length=200)
    user_id: str | None = Field(default=None, max_length=200)


class QueryResponse(BaseModel):
    answer: str
    # Lets a user reporting a bad answer be matched to the trace that produced
    # it. Null when tracing is disabled or failed to start.
    trace_id: str | None = None


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
    status: str | None = None
    data_source: str


class TransactionListResponse(BaseModel):
    data: list[TransactionRecord]
    total: int


class StatusCountsResponse(BaseModel):
    received: int
    flagged: int
    embedding: int
    embedded: int


class ServiceHealth(BaseModel):
    name: str
    label: str
    status: str  # "up" | "down"


class ServiceHealthResponse(BaseModel):
    services: list[ServiceHealth]