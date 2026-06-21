# Fraud RAG

Real-time credit-card fraud monitoring with a retrieval-augmented LLM agent for
investigating flagged transactions in natural language.

Transactions land on a Kafka topic, get persisted to Postgres, and are
asynchronously embedded into `pgvector` for semantic retrieval. An LLM agent
(LangGraph + MCP tools) answers analyst questions like *"any anomalous
transactions over 1000 EUR in the last hour?"* by pulling similar past
transactions from the vector store and scoring their features against a
correlation model. A React dashboard exposes both a live transaction
monitor and a chat interface to the agent.

## Architecture

```
                         ┌──────────────────────┐
  (external producer) ─▶ │   Kafka topic         │
                         │ transaction-decisions │
                         └──────────┬────────────┘
                                    │
                            ┌───────▼────────┐
                            │ FraudTransaction│   services/consumer
                            │   Consumer      │   → writes to `transactions`
                            └───────┬────────┘
                                    │
                            ┌───────▼────────┐
                            │ EmbeddingWorker │   services/embedder
                            │ (poll + encode) │   → writes to `transaction_embeddings`
                            └───────┬────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                      Postgres + pgvector                │
        └───────────────────────────┬───────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       │                            │                             │
┌──────▼───────┐           ┌────────▼────────┐           ┌────────▼────────┐
│ mcp-repository│           │  mcp-analysis   │           │     rag-api     │
│ context_lookup│◀──tools──│interpret_fraud_  │──tools──▶│ FastAPI + agent │
│  (vector kNN) │           │    features     │           │ (LangGraph)     │
└──────────────┘           └─────────────────┘           └────────┬────────┘
                                                                    │
                                                          ┌─────────▼─────────┐
                                                          │   React dashboard  │
                                                          └────────────────────┘
```

The two MCP servers are plain [FastMCP](https://github.com/jlowin/fastmcp)
processes exposing tools over SSE; the agent (`services/agent/graph.py`) is a
small [LangGraph](https://github.com/langchain-ai/langgraph) loop that binds
those tools to a chat model and lets it decide when to call them.

## Components

| Path | Responsibility |
|---|---|
| `services/consumer/consumer.py` | Kafka consumer → validates payloads against `TransactionCanonical` → idempotent insert into `transactions` |
| `services/embedder/` | Polls for transactions without an embedding, encodes them with a SentenceTransformer model, writes to `transaction_embeddings` |
| `services/repository/` | SQLAlchemy data access layer; all repositories share one process-wide engine (`services/repository/base.py`) |
| `services/tool/rag_tool.py` | k-NN similarity search over `transaction_embeddings` (pgvector `l2_distance`) |
| `services/mcp_server/repository_server.py` | Exposes `context_lookup` (the RAG tool above) as an MCP tool |
| `services/mcp_server/analysis_server.py` | Exposes `interpret_fraud_features`, which scores V-features against a precomputed correlation map (`config/*.yaml: correlation_analysis`) |
| `services/agent/` | LLM client (`agent.py`) + LangGraph workflow that wires the chat model to both MCP tool servers (`graph.py`) |
| `services/api/main.py` | FastAPI app: `/ask` (agent Q&A), `/transactions/timeseries` (dashboard data), `/health`. Also bootstraps the consumer and embedder as background threads in dev/single-process mode |
| `frontend/` | React (Vite) dashboard: live transaction chart + chat UI against `/ask` |
| `database/model.py`, `alembic/` | SQLAlchemy models and migrations for `transactions` / `transaction_embeddings` |
| `shared/config_loader.py` | Loads `config/application*.yaml` based on `APP_ENV`, expanding `${VAR}` placeholders from the environment |

> The transaction **producer** (whatever publishes to the `transaction-decisions`
> Kafka topic and serves the dashboard's "inject" endpoint) is not part of this
> repository — it's referenced only via `dashboard.inject_url` in config.

## Data flow contract

Messages on the Kafka topic are JSON matching `schemas.transaction.TransactionCanonical`:

```json
{
  "transaction_id": "uuid (optional, generated if absent)",
  "event_time_seconds": 123456,
  "event_timestamp": "2026-06-20T10:00:00",
  "amount": 100.50,
  "features": {"V1": 1.2, "V2": -0.5, "...": "..."},
  "is_fraud": false,
  "data_source": "kaggle-creditcard"
}
```

## Running it

### Docker Compose (full stack)

```bash
cp .env.example .env   # fill in POSTGRES_* and OPENAI_API_KEY if needed
docker compose up --build
```

This brings up Postgres (with pgvector), Ollama (pulls `llama3` on first
start), both MCP servers, the RAG API, and the React dashboard. Run
migrations against the running database with:

```bash
alembic upgrade head
```

| Service | URL |
|---|---|
| RAG API | http://localhost:8002 |
| Dashboard | http://localhost:8080 |
| mcp-repository | http://localhost:8003/sse |
| mcp-analysis | http://localhost:8004/sse |

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
./scripts.sh start   # consumer + embedder worker (background) + API (foreground, :8000)
```

Or run any single service directly, e.g.:

```bash
python -m services.consumer.consumer
python -m services.embedder.worker
python -m services.mcp_server.repository_server
python -m services.mcp_server.analysis_server
```

`config/application.yaml` (loaded by default, i.e. when `APP_ENV` is unset)
points everything at `localhost`, matching this workflow.

### Frontend (dashboard)

```bash
cd frontend
npm install
cp .env.example .env.local   # set VITE_API_BASE_URL if the API isn't on :8000
npm run dev                  # http://localhost:5173
```

The dashboard is a Vite + React + TypeScript app (`frontend/`) with two
views: **Agent**, a chat UI against `POST /ask`, and **Monitor**, the live
transaction chart against `GET /transactions/timeseries` (plus the inject
control, if `VITE_INJECT_URL` is set). It talks to the API directly from the
browser, so the API needs `CORS_ORIGINS` to include the dashboard's origin
(see `services/api/main.py`).

## Configuration

Config is environment-layered YAML, picked by `APP_ENV`:

- `APP_ENV` unset → `config/application.yaml` (local dev defaults)
- `APP_ENV=prod` → `config/application-prod.yaml`

Values may reference environment variables with `${VAR}` syntax (expanded
against `os.environ` before YAML parsing), which is how `application-prod.yaml`
picks up `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` /
`OPENAI_API_KEY` injected by `docker-compose.yaml`.

Key sections:

- `database` — connection URL + embedder batch size
- `kafka` — bootstrap servers, topic, consumer group
- `embedding` — sentence-transformer model name + vector dimension (must match the `Vector(...)` size in `database/model.py`)
- `llm` — provider/base_url/model for the agent's chat client (defaults to a local Ollama or LM Studio endpoint; swap to OpenAI by uncommenting the block in `application-prod.yaml`)
- `correlation_analysis` — feature→correlation map and risk thresholds used by `interpret_fraud_features`
- `mcp_servers` — SSE URLs the agent connects to

## API

- `GET /health` — liveness check
- `POST /ask` — `{"prompt": str, "top_k": int}` → `{"answer": str}`. Runs the LangGraph agent end-to-end (vector search + feature analysis as needed).
- `GET /transactions/timeseries?start=...&end=...` — per-minute transaction/fraud/normal counts for the dashboard chart.

## Database

```
transactions(transaction_id PK, event_time_seconds, event_timestamp [indexed],
             amount, is_fraud, features JSONB, data_source, created_at)
transaction_embeddings(transaction_id PK/FK → transactions, embedding vector(384),
                        embedding_text, embedding_model, created_at)
```

Migrations live in `alembic/versions/`. Apply with `alembic upgrade head`; the
URL is sourced from the same `shared/config_loader` the app uses, so `APP_ENV`
applies to migrations too.

## Tests

```bash
pytest test/
```
