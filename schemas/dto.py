from pydantic import BaseModel


class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5


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