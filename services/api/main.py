import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.api.rag_service import RAGQueryEngine
from services.consumer.consumer import FraudTransactionConsumer
from services.embedder.worker import EmbeddingWorker
from shared.config_loader import ConfigLoader

consumer = FraudTransactionConsumer()
config_loader = ConfigLoader()
embedder = EmbeddingWorker()

class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str

cfg = config_loader.load()
engine = RAGQueryEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = threading.Event()
    consumer_thread = threading.Thread(target=consumer.start, daemon=True)
    consumer_thread.start()

    embedder_thread = threading.Thread(target=embedder.start, daemon=True)
    embedder_thread.start()

    yield

    stop_event.set()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=QueryResponse)
async def ask_retrieval_augmented_generation(query: QueryRequest):
    try:
        answer = engine.ask(query.prompt, query.top_k)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
