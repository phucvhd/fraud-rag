import threading
from unittest.mock import patch, MagicMock

from services.embedder.worker import EmbeddingWorker


@patch("services.embedder.worker.EmbeddingProcessor")
@patch("services.embedder.worker.TransactionEmbeddingRepository")
@patch("services.embedder.worker.config_loader")
def test_embedding_worker_init(mock_config_loader, mock_repo, mock_processor):
    mock_config_loader.load.return_value = MagicMock()
    model = MagicMock()

    worker = EmbeddingWorker(model)

    assert worker.repo is mock_repo.return_value
    assert worker.processor is mock_processor.return_value
    mock_processor.assert_called_once_with(model)


@patch("services.embedder.worker.EmbeddingProcessor")
@patch("services.embedder.worker.TransactionEmbeddingRepository")
@patch("services.embedder.worker.config_loader")
def test_embedding_worker_processes_and_saves(mock_config_loader, mock_repo, mock_processor):
    cfg = MagicMock()
    cfg.database.batch_size = 10
    cfg.embedding.model_name = "test-model"
    mock_config_loader.load.return_value = cfg

    job = {
        "transaction_id": "11111111-1111-1111-1111-111111111111",
        "amount": 100.0,
        "features": {"V1": 0.5},
        "is_fraud": False,
    }
    repo = mock_repo.return_value
    repo.fetch_pending.return_value = [job]
    mock_processor.return_value.create_embedding.return_value = ([0.1, 0.2], "embedding text")

    stop_event = threading.Event()
    # Break the loop right after the first job is saved.
    repo.save.side_effect = lambda embedding: stop_event.set()

    worker = EmbeddingWorker(MagicMock())
    worker.start(stop_event)

    mock_processor.return_value.create_embedding.assert_called_once_with(100.0, {"V1": 0.5}, False)
    repo.save.assert_called_once()
    saved = repo.save.call_args.args[0]
    assert str(saved.transaction_id) == job["transaction_id"]
    assert saved.embedding == [0.1, 0.2]
    assert saved.embedding_text == "embedding text"
    assert saved.embedding_model == "test-model"
