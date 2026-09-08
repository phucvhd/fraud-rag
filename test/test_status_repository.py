from datetime import datetime
from unittest.mock import patch, MagicMock

from services.repository.status_repository import TransactionStatusRepository


def _repo(mock_config_loader, mock_get_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    return TransactionStatusRepository(), mock_conn


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_mark_received_upserts_all_ids_in_one_call(mock_config_loader, mock_get_engine):
    repo, mock_conn = _repo(mock_config_loader, mock_get_engine)

    repo.mark_received(["id-1", "id-2"])

    mock_conn.execute.assert_called_once()
    stmt, params = mock_conn.execute.call_args[0]
    assert "received_at" in str(stmt)
    assert [p["transaction_id"] for p in params] == ["id-1", "id-2"]
    assert all(p["status"] == "received" for p in params)


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_mark_embedding_and_mark_embedded_use_distinct_columns(mock_config_loader, mock_get_engine):
    repo, mock_conn = _repo(mock_config_loader, mock_get_engine)

    repo.mark_embedding(["id-1"])
    embedding_stmt = mock_conn.execute.call_args[0][0]

    repo.mark_embedded(["id-1"])
    embedded_stmt = mock_conn.execute.call_args[0][0]

    assert "embedding_started_at" in str(embedding_stmt)
    assert "embedded_at" in str(embedded_stmt)


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_mark_many_empty_list_is_a_noop(mock_config_loader, mock_get_engine):
    repo, mock_conn = _repo(mock_config_loader, mock_get_engine)

    repo.mark_flagged([])

    mock_conn.execute.assert_not_called()


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_mark_many_swallows_db_errors(mock_config_loader, mock_get_engine):
    # Status tracking must never bubble up and crash the caller (Kafka
    # consumer / embedder loop) on a transient Postgres failure.
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_engine.begin.side_effect = RuntimeError("connection refused")

    repo = TransactionStatusRepository()
    repo.mark_received(["id-1"])  # should not raise


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_get_status_counts_fills_in_zero_for_missing_statuses(mock_config_loader, mock_get_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.all.return_value = [
        {"status": "received", "count": 2},
        {"status": "embedded", "count": 500},
    ]

    repo = TransactionStatusRepository()
    counts = repo.get_status_counts(datetime(2026, 9, 7), datetime(2026, 9, 8))

    assert counts == {"received": 2, "flagged": 0, "embedding": 0, "embedded": 500}
