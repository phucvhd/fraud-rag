import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from schemas.dto import QueryResponse, QueryRequest
from services.agent.agent import LLMAgent
from services.agent.graph import FraudInspectorGraph
from services.agent.sentence_transformer import SentenceTransformerModel
from services.consumer.consumer import FraudTransactionConsumer
from services.embedder.worker import EmbeddingWorker
from shared.config_loader import ConfigLoader

consumer = FraudTransactionConsumer()
config_loader = ConfigLoader()
sentence_transformer_model = SentenceTransformerModel()
embedder = EmbeddingWorker(sentence_transformer_model)

cfg = config_loader.load()
agent = LLMAgent()
inspector = FraudInspectorGraph(agent)

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
async def ask_anomaly_analysis(query: QueryRequest):
    try:
        print("Received anomaly analysis request")
        answer = await inspector.run(query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
