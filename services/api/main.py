import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from schemas.dto import (
    QueryRequest,
    QueryResponse,
    ServiceHealthResponse,
    StatusCountsResponse,
    TimeseriesBucket,
    TimeseriesResponse,
    TransactionListResponse,
    TransactionRecord,
)
from services.agent.agent import LLMAgent
from services.agent.graph import FraudInspectorGraph
from services.agent.sentence_transformer import SentenceTransformerModel
from services.consumer.consumer import FraudTransactionConsumer
from services.embedder.worker import EmbeddingWorker
from services.health.health_checker import HealthChecker
from services.repository.status_repository import TransactionStatusRepository
from services.repository.transaction_canonical_repository import TransactionCanonicalRepository
from shared.config_loader import config_loader
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
    app.state.status_repo = TransactionStatusRepository()
    app.state.health_checker = HealthChecker(config_loader.load(), app.state.transaction_repo.engine)

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
        result = await request.app.state.inspector.run(query)
        logger.info("Answered analysis request, trace_id=%s", result.trace_id)
        return QueryResponse(answer=result.answer, trace_id=result.trace_id)
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


@app.get("/transactions", response_model=TransactionListResponse)
async def get_transactions(
    request: Request,
    start: datetime = Query(...),
    end: datetime = Query(...),
    limit: int = Query(default=50, gt=0, le=200),
    offset: int = Query(default=0, ge=0),
    is_fraud: bool | None = Query(default=None),
    pipeline_status: str | None = Query(default=None, pattern="^(received|flagged|embedding|embedded)$"),
    search: str | None = Query(default=None, max_length=100),
    sort_by: str = Query(default="time", pattern="^(time|amount|status|risk)$"),
    sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
):
    try:
        rows, total = request.app.state.transaction_repo.get_transactions(
            start,
            end,
            limit=limit,
            offset=offset,
            is_fraud=is_fraud,
            pipeline_status=pipeline_status,
            search=search,
            sort_by=sort_by,
            sort_dir=sort_dir,
        )
        return TransactionListResponse(data=[TransactionRecord(**r) for r in rows], total=total)
    except Exception:
        logger.exception("Transaction list query failed")
        raise HTTPException(status_code=500, detail="Failed to fetch transactions.")


@app.get("/transactions/status-counts", response_model=StatusCountsResponse)
async def get_status_counts(
    request: Request,
    start: datetime = Query(...),
    end: datetime = Query(...),
):
    try:
        counts = request.app.state.status_repo.get_status_counts(start, end)
        return StatusCountsResponse(**counts)
    except Exception:
        logger.exception("Status counts query failed")
        raise HTTPException(status_code=500, detail="Failed to fetch status counts.")


@app.get("/health/dependencies", response_model=ServiceHealthResponse)
async def get_dependency_health(request: Request):
    # Deliberately has no try/except around the whole body: check_all() never
    # raises (each individual check swallows its own errors and reports
    # "down"), so a 500 here would mean a real bug in the checker itself.
    services = await request.app.state.health_checker.check_all()
    return ServiceHealthResponse(services=services)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
