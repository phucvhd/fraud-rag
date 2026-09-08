import threading
from unittest.mock import patch, MagicMock

from services.embedder.worker import EmbeddingWorker


@patch("services.embedder.worker.EmbeddingProcessor")
@patch("services.embedder.worker.TransactionStatusRepository")
@patch("services.embedder.worker.TransactionEmbeddingRepository")
@patch("services.embedder.worker.config_loader")
def test_embedding_worker_init(mock_config_loader, mock_repo, mock_status_repo, mock_processor):
    mock_config_loader.load.return_value = MagicMock()
    model = MagicMock()

    worker = EmbeddingWorker(model)

    assert worker.repo is mock_repo.return_value
    assert worker.status_repo is mock_status_repo.return_value
    assert worker.processor is mock_processor.return_value
    mock_processor.assert_called_once_with(model)


@patch("services.embedder.worker.EmbeddingProcessor")
@patch("services.embedder.worker.TransactionStatusRepository")
@patch("services.embedder.worker.TransactionEmbeddingRepository")
@patch("services.embedder.worker.config_loader")
def test_embedding_worker_processes_and_saves(mock_config_loader, mock_repo, mock_status_repo, mock_processor):
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
    mock_processor.return_value.create_embeddings.return_value = [([0.1, 0.2], "embedding text")]

    status_repo = mock_status_repo.return_value
    stop_event = threading.Event()
    # Break the loop right after the batch is saved.
    repo.save_many.side_effect = lambda embeddings: stop_event.set()

    worker = EmbeddingWorker(MagicMock())
    worker.start(stop_event)

    mock_processor.return_value.create_embeddings.assert_called_once_with([job])
    repo.save_many.assert_called_once()
    saved = repo.save_many.call_args.args[0]
    assert len(saved) == 1
    assert str(saved[0].transaction_id) == job["transaction_id"]
    assert saved[0].embedding == [0.1, 0.2]
    assert saved[0].embedding_text == "embedding text"
    assert saved[0].embedding_model == "test-model"

    status_repo.mark_embedding.assert_called_once_with([job["transaction_id"]])
    status_repo.mark_embedded.assert_called_once_with([job["transaction_id"]])
