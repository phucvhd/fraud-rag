import logging
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

logger = logging.getLogger(__name__)

inspector: FraudInspectorGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inspector

    sentence_transformer_model = SentenceTransformerModel()
    consumer = FraudTransactionConsumer()
    embedder = EmbeddingWorker(sentence_transformer_model)
    agent = LLMAgent()
    inspector = FraudInspectorGraph(agent)

    stop_event = threading.Event()

    consumer_thread = threading.Thread(
        target=consumer.start, args=(stop_event,), daemon=True
    )
    embedder_thread = threading.Thread(
        target=embedder.start, args=(stop_event,), daemon=True
    )

    consumer_thread.start()
    embedder_thread.start()

    yield

    stop_event.set()
    consumer_thread.join(timeout=10)
    embedder_thread.join(timeout=10)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=QueryResponse)
async def ask_anomaly_analysis(query: QueryRequest):
    try:
        logger.info("Received anomaly analysis request")
        answer = await inspector.run(query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
