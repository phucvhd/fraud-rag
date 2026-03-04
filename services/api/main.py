import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from schemas.dto import QueryResponse, QueryRequest
from services.agent.agent import LLMAgent
from services.api.rag_service import RAGQueryEngine
from services.consumer.consumer import FraudTransactionConsumer
from services.embedder.worker import EmbeddingWorker
from services.tool.fraud_agent_tool import FraudInspector
from shared.config_loader import ConfigLoader

consumer = FraudTransactionConsumer()
config_loader = ConfigLoader()
embedder = EmbeddingWorker()

cfg = config_loader.load()
agent = LLMAgent()
engine = RAGQueryEngine(agent)
inspector = FraudInspector(cfg, agent, engine)

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
        answer = await inspector.run(query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
