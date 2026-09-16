import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from services.health.health_checker import HealthChecker


def _cfg():
    cfg = MagicMock()
    cfg.kafka.bootstrap_servers = "localhost:9092"
    cfg.dashboard.fraud_detection_base_url = "http://localhost:8000"
    cfg.llm.base_url = "http://localhost:1234/v1"
    cfg.llm.api_key = "test-key"
    cfg.mcp_servers.repository.url = "http://localhost:8003/sse"
    return cfg


def _mock_http_client(mock_response=None, side_effect=None):
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response, side_effect=side_effect)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def test_check_postgres_up():
    checker = HealthChecker(_cfg(), MagicMock())

    result = asyncio.run(checker._check_postgres())

    assert result == {"name": "postgres", "label": "Postgres", "status": "up"}


def test_check_postgres_down():
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("connection refused")
    checker = HealthChecker(_cfg(), engine)

    result = asyncio.run(checker._check_postgres())

    assert result == {"name": "postgres", "label": "Postgres", "status": "down"}


@patch("services.health.health_checker.AdminClient")
def test_check_kafka_up(mock_admin_client):
    metadata = MagicMock()
    metadata.brokers = {1: MagicMock()}
    mock_admin_client.return_value.list_topics.return_value = metadata
    checker = HealthChecker(_cfg(), MagicMock())

    result = asyncio.run(checker._check_kafka())

    assert result == {"name": "kafka", "label": "Kafka", "status": "up"}


@patch("services.health.health_checker.AdminClient")
def test_check_kafka_down_when_no_brokers(mock_admin_client):
    metadata = MagicMock()
    metadata.brokers = {}
    mock_admin_client.return_value.list_topics.return_value = metadata
    checker = HealthChecker(_cfg(), MagicMock())

    result = asyncio.run(checker._check_kafka())

    assert result == {"name": "kafka", "label": "Kafka", "status": "down"}


def test_check_http_up_on_2xx():
    checker = HealthChecker(_cfg(), MagicMock())
    mock_response = MagicMock(status_code=200)

    with patch("services.health.health_checker.httpx.AsyncClient", return_value=_mock_http_client(mock_response)):
        result = asyncio.run(checker._check_http("svc", "Service", "http://x/health"))

    assert result == {"name": "svc", "label": "Service", "status": "up"}


def test_check_http_down_on_5xx():
    checker = HealthChecker(_cfg(), MagicMock())
    mock_response = MagicMock(status_code=503)

    with patch("services.health.health_checker.httpx.AsyncClient", return_value=_mock_http_client(mock_response)):
        result = asyncio.run(checker._check_http("svc", "Service", "http://x/health"))

    assert result == {"name": "svc", "label": "Service", "status": "down"}


def test_check_http_down_on_connection_error():
    checker = HealthChecker(_cfg(), MagicMock())
    client = _mock_http_client(side_effect=ConnectionError("refused"))

    with patch("services.health.health_checker.httpx.AsyncClient", return_value=client):
        result = asyncio.run(checker._check_http("svc", "Service", "http://x/health"))

    assert result == {"name": "svc", "label": "Service", "status": "down"}


def test_check_tcp_up():
    checker = HealthChecker(_cfg(), MagicMock())
    mock_writer = MagicMock()
    mock_writer.wait_closed = AsyncMock()

    async def fake_open_connection(host, port):
        return MagicMock(), mock_writer

    with patch("services.health.health_checker.asyncio.open_connection", side_effect=fake_open_connection):
        result = asyncio.run(checker._check_tcp("mcp_repository", "MCP Repository", "http://localhost:8003/sse"))

    assert result == {"name": "mcp_repository", "label": "MCP Repository", "status": "up"}
    mock_writer.close.assert_called_once()


def test_check_tcp_down_on_connection_refused():
    checker = HealthChecker(_cfg(), MagicMock())

    async def fake_open_connection(host, port):
        raise ConnectionRefusedError("refused")

    with patch("services.health.health_checker.asyncio.open_connection", side_effect=fake_open_connection):
        result = asyncio.run(checker._check_tcp("mcp_repository", "MCP Repository", "http://localhost:8003/sse"))

    assert result == {"name": "mcp_repository", "label": "MCP Repository", "status": "down"}


def test_check_all_gathers_every_dependency():
    checker = HealthChecker(_cfg(), MagicMock())
    checker._check_postgres = AsyncMock(return_value={"name": "postgres", "label": "Postgres", "status": "up"})
    checker._check_kafka = AsyncMock(return_value={"name": "kafka", "label": "Kafka", "status": "up"})
    checker._check_fraud_detection = AsyncMock(
        return_value={"name": "fraud_detection", "label": "Fraud Detection", "status": "down"}
    )
    checker._check_tcp = AsyncMock(return_value={"name": "mcp", "label": "MCP", "status": "up"})
    checker._check_llm = AsyncMock(return_value={"name": "llm", "label": "LLM", "status": "up"})

    results = asyncio.run(checker.check_all())

    assert len(results) == 5
    names = [r["name"] for r in results]
    assert names.count("postgres") == 1
    assert names.count("kafka") == 1
    assert names.count("fraud_detection") == 1
