import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from schemas.dto import QueryRequest, QueryResponse, TimeseriesBucket, TimeseriesResponse
from services.agent.agent import LLMAgent
from services.agent.graph import FraudInspectorGraph
from services.agent.sentence_transformer import SentenceTransformerModel
from services.consumer.consumer import FraudTransactionConsumer
from services.embedder.worker import EmbeddingWorker
from services.repository.transaction_canonical_repository import TransactionCanonicalRepository
from shared.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sentence_transformer_model = SentenceTransformerModel()
    consumer = FraudTransactionConsumer()
    embedder = EmbeddingWorker(sentence_transformer_model)
    agent = LLMAgent()

    app.state.inspector = FraudInspectorGraph(agent)
    app.state.transaction_repo = TransactionCanonicalRepository()

    stop_event = threading.Event()
    consumer_thread = threading.Thread(target=consumer.start, args=(stop_event,), daemon=True)
    embedder_thread = threading.Thread(target=embedder.start, args=(stop_event,), daemon=True)
    consumer_thread.start()
    embedder_thread.start()

    yield

    stop_event.set()
    consumer_thread.join(timeout=10)
    embedder_thread.join(timeout=10)


app = FastAPI(lifespan=lifespan)

cors_origins = [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=QueryResponse)
async def ask_anomaly_analysis(query: QueryRequest, request: Request):
    try:
        logger.info("Received anomaly analysis request")
        answer = await request.app.state.inspector.run(query)
        return QueryResponse(answer=answer)
    except Exception:
        logger.exception("Anomaly analysis request failed")
        raise HTTPException(status_code=500, detail="Failed to process the analysis request.")


@app.get("/transactions/timeseries", response_model=TimeseriesResponse)
async def get_transaction_timeseries(
    request: Request,
    start: datetime = Query(...),
    end: datetime = Query(...),
):
    try:
        rows = request.app.state.transaction_repo.get_timeseries(start, end)
        total_tx = sum(r["transactions"] for r in rows)
        total_fraud = sum(r["fraud"] for r in rows)
        total_normal = sum(r["normal"] for r in rows)
        return TimeseriesResponse(
            data=[TimeseriesBucket(**r) for r in rows],
            total_transactions=total_tx,
            total_fraud=total_fraud,
            total_normal=total_normal,
        )
    except Exception:
        logger.exception("Timeseries query failed")
        raise HTTPException(status_code=500, detail="Failed to fetch transaction timeseries.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
