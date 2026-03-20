from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


def _make_client(inspector_run=None):
    """Return a TestClient with all heavy lifespan dependencies mocked."""
    mock_inspector = MagicMock()
    mock_inspector.run = AsyncMock(return_value="Test answer") if inspector_run is None else AsyncMock(side_effect=inspector_run)

    with (
        patch("services.api.main.SentenceTransformerModel"),
        patch("services.api.main.FraudTransactionConsumer"),
        patch("services.api.main.EmbeddingWorker"),
        patch("services.api.main.LLMAgent"),
        patch("services.api.main.FraudInspectorGraph", return_value=mock_inspector),
        patch("services.api.main.threading.Thread"),
    ):
        from services.api.main import app
        return TestClient(app), mock_inspector


def test_health_check():
    client, _ = _make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_endpoint_success():
    client, _ = _make_client()
    response = client.post("/ask", json={"prompt": "test query", "top_k": 3})
    assert response.status_code == 200
    assert response.json() == {"answer": "Test answer"}


def test_ask_endpoint_exception():
    client, _ = _make_client(inspector_run=Exception("Test error"))
    response = client.post("/ask", json={"prompt": "test query"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Test error"}
