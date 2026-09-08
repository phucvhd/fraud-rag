from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from services.api.main import app


def _client_with_inspector(inspector_run=None):
    """TestClient with app.state wired directly.

    The endpoints read ``request.app.state.inspector`` / ``transaction_repo``,
    which the lifespan normally populates. TestClient without a context manager
    does not run the lifespan, so we set the state ourselves and avoid spinning
    up the real consumer/embedder threads and heavy models.
    """
    mock_inspector = MagicMock()
    if inspector_run is None:
        mock_inspector.run = AsyncMock(return_value="Test answer")
    else:
        mock_inspector.run = AsyncMock(side_effect=inspector_run)

    app.state.inspector = mock_inspector
    app.state.transaction_repo = MagicMock()
    app.state.status_repo = MagicMock()
    app.state.health_checker = MagicMock()
    app.state.health_checker.check_all = AsyncMock(return_value=[])
    return TestClient(app), mock_inspector


def test_health_check():
    client, _ = _client_with_inspector()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_endpoint_success():
    client, _ = _client_with_inspector()
    response = client.post("/ask", json={"prompt": "test query", "top_k": 3})
    assert response.status_code == 200
    assert response.json() == {"answer": "Test answer"}


def test_ask_endpoint_exception():
    client, _ = _client_with_inspector(inspector_run=Exception("Test error"))
    response = client.post("/ask", json={"prompt": "test query"})
    assert response.status_code == 500
    # Internal error details are intentionally hidden behind a generic message.
    assert response.json() == {"detail": "Failed to process the analysis request."}


def test_get_transactions_success():
    client, _ = _client_with_inspector()
    row = {
        "transaction_id": "11111111-1111-1111-1111-111111111111",
        "event_timestamp": "2026-09-07T12:00:00",
        "amount": 42.5,
        "is_fraud": False,
        "data_source": "test",
    }
    app.state.transaction_repo.get_transactions.return_value = ([row], 1)

    response = client.get(
        "/transactions",
        params={"start": "2026-09-07T00:00:00", "end": "2026-09-08T00:00:00"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["data"][0]["transaction_id"] == row["transaction_id"]
    assert body["data"][0]["amount"] == 42.5


def test_get_transactions_passes_pipeline_status_filter():
    client, _ = _client_with_inspector()
    app.state.transaction_repo.get_transactions.return_value = ([], 0)

    response = client.get(
        "/transactions",
        params={"start": "2026-09-07T00:00:00", "end": "2026-09-08T00:00:00", "pipeline_status": "embedding"},
    )

    assert response.status_code == 200
    kwargs = app.state.transaction_repo.get_transactions.call_args.kwargs
    assert kwargs["pipeline_status"] == "embedding"


def test_get_transactions_rejects_invalid_pipeline_status():
    client, _ = _client_with_inspector()

    response = client.get(
        "/transactions",
        params={"start": "2026-09-07T00:00:00", "end": "2026-09-08T00:00:00", "pipeline_status": "bogus"},
    )

    assert response.status_code == 422


def test_get_transactions_exception():
    client, _ = _client_with_inspector()
    app.state.transaction_repo.get_transactions.side_effect = Exception("db down")

    response = client.get(
        "/transactions",
        params={"start": "2026-09-07T00:00:00", "end": "2026-09-08T00:00:00"},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to fetch transactions."}


def test_get_status_counts_success():
    client, _ = _client_with_inspector()
    app.state.status_repo.get_status_counts.return_value = {
        "received": 2,
        "flagged": 1,
        "embedding": 0,
        "embedded": 500,
    }

    response = client.get(
        "/transactions/status-counts",
        params={"start": "2026-09-07T00:00:00", "end": "2026-09-08T00:00:00"},
    )
    assert response.status_code == 200
    assert response.json() == {"received": 2, "flagged": 1, "embedding": 0, "embedded": 500}


def test_get_status_counts_exception():
    client, _ = _client_with_inspector()
    app.state.status_repo.get_status_counts.side_effect = Exception("db down")

    response = client.get(
        "/transactions/status-counts",
        params={"start": "2026-09-07T00:00:00", "end": "2026-09-08T00:00:00"},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to fetch status counts."}


def test_get_dependency_health_success():
    client, _ = _client_with_inspector()
    app.state.health_checker.check_all = AsyncMock(
        return_value=[
            {"name": "postgres", "label": "Postgres", "status": "up"},
            {"name": "kafka", "label": "Kafka", "status": "down"},
        ]
    )

    response = client.get("/health/dependencies")

    assert response.status_code == 200
    assert response.json() == {
        "services": [
            {"name": "postgres", "label": "Postgres", "status": "up"},
            {"name": "kafka", "label": "Kafka", "status": "down"},
        ]
    }
