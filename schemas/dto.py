from pydantic import BaseModel

class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str