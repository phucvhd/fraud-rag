import pytest
from unittest.mock import patch, MagicMock
from services.embedder.worker import EmbeddingWorker

@patch("services.embedder.worker.create_engine")
@patch("services.embedder.worker.config_loader")
@patch("services.embedder.worker.EmbeddingProcessor")
def test_embedding_worker_init(mock_processor, mock_config_loader, mock_create_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    worker = EmbeddingWorker()

    mock_create_engine.assert_called_once_with("sqlite:///:memory:")
    assert worker.processor == mock_processor.return_value

@patch("services.embedder.worker.select")
@patch("services.embedder.worker.create_engine")
@patch("services.embedder.worker.config_loader")
@patch("services.embedder.worker.EmbeddingProcessor")
def test_embedding_worker_fetch_pending(mock_processor, mock_config_loader, mock_create_engine, mock_select):
    mock_config = MagicMock()
    mock_config.database.batch_size = 10
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_create_engine.return_value = mock_engine

    mock_records = [{"transaction_id": "1", "amount": 100, "features": {}}]
    mock_conn.execute.return_value.mappings.return_value.all.return_value = mock_records

    worker = EmbeddingWorker()
    jobs = worker._fetch_pending()

    assert jobs == mock_records
    mock_conn.execute.assert_called_once()

@patch("services.embedder.worker.insert")
@patch("services.embedder.worker.create_engine")
@patch("services.embedder.worker.config_loader")
@patch("services.embedder.worker.EmbeddingProcessor")
def test_embedding_worker_save_vector(mock_processor, mock_config_loader, mock_create_engine, mock_insert):
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn
    mock_create_engine.return_value = mock_engine

    mock_stmt = MagicMock()
    mock_insert.return_value.values.return_value.on_conflict_do_nothing.return_value = mock_stmt

    worker = EmbeddingWorker()
    worker._save_vector("t_id", [0.1], "text")

    mock_conn.execute.assert_called_once_with(mock_stmt)
