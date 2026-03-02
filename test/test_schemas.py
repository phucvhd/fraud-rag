import pytest
from datetime import datetime
from uuid import UUID
from schemas.transaction import TransactionBase, TransactionCanonical, TransactionEmbedding

def test_transaction_base_valid():
    data = {"event_time_seconds": 12345, "amount": 100.5, "features": {"f1": 0.5}, "data_source": "test"}
    tx = TransactionBase(**data)
    assert isinstance(tx.transaction_id, UUID)
    assert tx.event_time_seconds == 12345
    assert tx.amount == 100.5
    assert tx.features == {"f1": 0.5}
    assert tx.is_fraud is False
    assert tx.data_source == "test"

def test_transaction_base_invalid_amount():
    data = {"event_time_seconds": 12345, "amount": 0, "features": {"f1": 0.5}, "data_source": "test"}
    with pytest.raises(ValueError):
        TransactionBase(**data)

def test_transaction_canonical():
    data = {"event_time_seconds": 12345, "amount": 100.5, "features": {"f1": 0.5}, "data_source": "test", "event_timestamp": "2023-01-01T12:00:00"}
    tx = TransactionCanonical(**data)
    assert isinstance(tx.event_timestamp, datetime)
    assert isinstance(tx.created_at, datetime)

def test_transaction_canonical_datetime():
    data = {"event_time_seconds": 12345, "amount": 100.5, "features": {"f1": 0.5}, "data_source": "test", "event_timestamp": datetime(2023, 1, 1, 12, 0, 0)}
    tx = TransactionCanonical(**data)
    assert isinstance(tx.event_timestamp, datetime)

def test_transaction_embedding():
    data = {"transaction_id": "123e4567-e89b-12d3-a456-426614174000", "embedding_text": "test text", "embedding_model": "test_model"}
    tx = TransactionEmbedding(**data)
    assert isinstance(tx.transaction_id, UUID)
    assert tx.embedding is None
    assert tx.embedding_text == "test text"
    assert tx.embedding_model == "test_model"
    assert isinstance(tx.created_at, datetime)
