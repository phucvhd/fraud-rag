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
