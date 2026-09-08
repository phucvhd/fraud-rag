from datetime import datetime
from unittest.mock import patch, MagicMock

from services.repository.transaction_canonical_repository import TransactionRepository
from schemas.transaction import TransactionCanonical


# BaseRepository resolves its engine via services.repository.base.get_engine using
# the URL from config_loader, so both are patched there (not on the concrete repo).
@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_repository_insert_if_not_exists(mock_config_loader, mock_get_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    repo = TransactionRepository()

    tx = TransactionCanonical(
        event_time_seconds=123,
        amount=10.5,
        features={"a": 1.0},
        data_source="test",
        event_timestamp=datetime(2023, 1, 1),
    )

    repo.insert_if_not_exists(tx)

    mock_get_engine.assert_called_once_with("sqlite:///:memory:")
    mock_conn.execute.assert_called_once()


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_repository_get_transactions(mock_config_loader, mock_get_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    row = {
        "transaction_id": "11111111-1111-1111-1111-111111111111",
        "event_timestamp": datetime(2026, 9, 7, 12, 0, 0),
        "amount": 42.5,
        "is_fraud": False,
        "data_source": "test",
    }
    rows_result = MagicMock()
    rows_result.mappings.return_value.all.return_value = [row]
    count_result = MagicMock()
    count_result.scalar_one.return_value = 1
    mock_conn.execute.side_effect = [rows_result, count_result]

    repo = TransactionRepository()
    rows, total = repo.get_transactions(datetime(2026, 9, 7), datetime(2026, 9, 8), limit=200)

    assert rows == [row]
    assert total == 1
    assert mock_conn.execute.call_count == 2


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_repository_get_transactions_with_filters(mock_config_loader, mock_get_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    rows_result = MagicMock()
    rows_result.mappings.return_value.all.return_value = []
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0
    mock_conn.execute.side_effect = [rows_result, count_result]

    repo = TransactionRepository()
    repo.get_transactions(
        datetime(2026, 9, 7),
        datetime(2026, 9, 8),
        limit=10,
        offset=20,
        is_fraud=True,
        search="abc",
        sort_by="amount",
        sort_dir="asc",
    )

    rows_query, rows_params = mock_conn.execute.call_args_list[0].args
    query_text = str(rows_query)
    assert "is_fraud = :is_fraud" in query_text
    assert "ILIKE" in query_text
    assert "ORDER BY amount ASC" in query_text
    assert rows_params == {
        "start": datetime(2026, 9, 7),
        "end": datetime(2026, 9, 8),
        "limit": 10,
        "offset": 20,
        "is_fraud": True,
        "search": "%abc%",
    }


@patch("services.repository.base.get_engine")
@patch("services.repository.base.config_loader")
def test_repository_get_transactions_unknown_sort_column_falls_back(mock_config_loader, mock_get_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    rows_result = MagicMock()
    rows_result.mappings.return_value.all.return_value = []
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0
    mock_conn.execute.side_effect = [rows_result, count_result]

    repo = TransactionRepository()
    repo.get_transactions(datetime(2026, 9, 7), datetime(2026, 9, 8), sort_by="'; DROP TABLE transactions; --")

    rows_query, _ = mock_conn.execute.call_args_list[0].args
    assert "ORDER BY event_timestamp DESC" in str(rows_query)
