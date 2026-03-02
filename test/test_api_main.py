import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

@patch("services.api.main.engine")
@patch("services.api.main.embedder")
@patch("services.api.main.consumer")
@patch("services.api.main.threading")
def test_health_check(mock_threading, mock_consumer, mock_embedder, mock_engine):
    from services.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("services.api.main.engine")
@patch("services.api.main.embedder")
@patch("services.api.main.consumer")
@patch("services.api.main.threading")
def test_ask_endpoint_success(mock_threading, mock_consumer, mock_embedder, mock_engine):
    from services.api.main import app
    mock_engine.ask.return_value = "Test answer"
    client = TestClient(app)
    
    response = client.post("/ask", json={"prompt": "test query", "top_k": 3})
    
    assert response.status_code == 200
    assert response.json() == {"answer": "Test answer"}
    mock_engine.ask.assert_called_once_with("test query", 3)

@patch("services.api.main.engine")
@patch("services.api.main.embedder")
@patch("services.api.main.consumer")
@patch("services.api.main.threading")
def test_ask_endpoint_exception(mock_threading, mock_consumer, mock_embedder, mock_engine):
    from services.api.main import app
    mock_engine.ask.side_effect = Exception("Test error")
    client = TestClient(app)
    
    response = client.post("/ask", json={"prompt": "test query"})
    
    assert response.status_code == 500
    assert response.json() == {"detail": "Test error"}
