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
