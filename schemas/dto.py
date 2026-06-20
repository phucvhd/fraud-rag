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