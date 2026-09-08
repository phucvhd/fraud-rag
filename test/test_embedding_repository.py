from datetime import datetime
from unittest.mock import patch, MagicMock

from schemas.transaction import TransactionEmbedding
from services.repository.embedding_repository import TransactionEmbeddingRepository


def _repo(mock_config_loader, mock_get_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    return TransactionEmbeddingRepository(), mock_conn


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_save_many_does_one_bulk_insert(mock_config_loader, mock_get_engine):
    repo, mock_conn = _repo(mock_config_loader, mock_get_engine)

    embeddings = [
        TransactionEmbedding(
            transaction_id="11111111-1111-1111-1111-111111111111",
            embedding=[0.1, 0.2],
            embedding_text="a",
            embedding_model="test-model",
            created_at=datetime(2026, 9, 7),
        ),
        TransactionEmbedding(
            transaction_id="22222222-2222-2222-2222-222222222222",
            embedding=[0.3, 0.4],
            embedding_text="b",
            embedding_model="test-model",
            created_at=datetime(2026, 9, 7),
        ),
    ]

    repo.save_many(embeddings)

    # One execute() for the whole batch, not one per row.
    mock_conn.execute.assert_called_once()


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_save_many_empty_list_is_a_noop(mock_config_loader, mock_get_engine):
    repo, mock_conn = _repo(mock_config_loader, mock_get_engine)

    repo.save_many([])

    mock_conn.execute.assert_not_called()
