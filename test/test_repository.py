import pytest
from unittest.mock import patch, MagicMock
from services.repository.transaction_canonical_repository import TransactionRepository
from schemas.transaction import TransactionCanonical
from datetime import datetime

@patch("services.repository.transaction_canonical_repository.create_engine")
@patch("services.repository.transaction_canonical_repository.config_loader")
def test_repository_insert_if_not_exists(mock_config_loader, mock_create_engine):
    mock_config = MagicMock()
    mock_config.database.url = "sqlite:///:memory:"
    mock_config_loader.load.return_value = mock_config

    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    repo = TransactionRepository()

    tx = TransactionCanonical(
        event_time_seconds=123,
        amount=10.5,
        features={"a": 1},
        data_source="test",
        event_timestamp=datetime(2023, 1, 1)
    )

    repo.insert_if_not_exists(tx)
    mock_conn.execute.assert_called_once()
