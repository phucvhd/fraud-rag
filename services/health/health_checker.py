import asyncio
import logging
from urllib.parse import urlparse

import httpx
from confluent_kafka.admin import AdminClient
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 2.0


class HealthChecker:
    """Checks reachability of every service this app depends on, so the
    dashboard can show one status row instead of the frontend having to know
    about Postgres/Kafka/MCP servers directly (which it can't reach anyway).
    """

    def __init__(self, cfg, engine: Engine):
        self.cfg = cfg
        self.engine = engine

    async def check_all(self) -> list[dict]:
        return list(
            await asyncio.gather(
                self._check_postgres(),
                self._check_kafka(),
                self._check_fraud_detection(),
                self._check_tcp("mcp_repository", "MCP Repository", self.cfg.mcp_servers.repository.url),
                self._check_llm(),
            )
        )

    async def _check_postgres(self) -> dict:
        try:
            await asyncio.wait_for(asyncio.to_thread(self._ping_postgres), timeout=_TIMEOUT_SECONDS)
            return {"name": "postgres", "label": "Postgres", "status": "up"}
        except Exception:
            logger.warning("Postgres health check failed", exc_info=True)
            return {"name": "postgres", "label": "Postgres", "status": "down"}

    def _ping_postgres(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    async def _check_kafka(self) -> dict:
        try:
            await asyncio.wait_for(asyncio.to_thread(self._ping_kafka), timeout=_TIMEOUT_SECONDS)
            return {"name": "kafka", "label": "Kafka", "status": "up"}
        except Exception:
            logger.warning("Kafka health check failed", exc_info=True)
            return {"name": "kafka", "label": "Kafka", "status": "down"}

    def _ping_kafka(self) -> None:
        admin = AdminClient({"bootstrap.servers": self.cfg.kafka.bootstrap_servers})
        metadata = admin.list_topics(timeout=_TIMEOUT_SECONDS)
        if not metadata.brokers:
            raise ConnectionError("No Kafka brokers reachable")

    async def _check_fraud_detection(self) -> dict:
        base_url = self.cfg.dashboard.fraud_detection_base_url
        return await self._check_http("fraud_detection", "Fraud Detection", f"{base_url.rstrip('/')}/health")

    async def _check_llm(self) -> dict:
        base_url = self.cfg.llm.base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {self.cfg.llm.api_key}"} if self.cfg.llm.api_key else {}
        return await self._check_http("llm", "LLM", f"{base_url}/models", headers=headers)

    async def _check_http(self, name: str, label: str, url: str, headers: dict | None = None) -> dict:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.get(url, headers=headers or {})
            # Any response at all (even a 4xx on an unexpected route) means the
            # process is up and answering; only connection-level failures count
            # as "down".
            status = "down" if resp.status_code >= 500 else "up"
            return {"name": name, "label": label, "status": status}
        except Exception:
            logger.warning("%s health check failed", label, exc_info=True)
            return {"name": name, "label": label, "status": "down"}

    async def _check_tcp(self, name: str, label: str, url: str) -> dict:
        # MCP servers speak SSE, not a plain HTTP health route — a successful
        # TCP handshake is enough to say "the process is up and listening".
        try:
            parsed = urlparse(url)
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(parsed.hostname, parsed.port), timeout=_TIMEOUT_SECONDS
            )
            writer.close()
            await writer.wait_closed()
            return {"name": name, "label": label, "status": "up"}
        except Exception:
            logger.warning("%s health check failed", label, exc_info=True)
            return {"name": name, "label": label, "status": "down"}
