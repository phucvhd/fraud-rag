import pytest
from unittest.mock import patch, MagicMock
from services.consumer.consumer import FraudTransactionConsumer
import json
from uuid import uuid4

@patch("services.consumer.consumer.Consumer")
@patch("services.consumer.consumer.config_loader")
@patch("services.consumer.consumer.TransactionRepository")
def test_consumer_init(mock_repo, mock_config_loader, mock_consumer_cls):
    mock_config = MagicMock()
    mock_config.kafka.bootstrap_servers = "locahost:9092"
    mock_config.kafka.group_id = "test_group"
    mock_config.kafka.topic = "test_topic"
    mock_config_loader.load.return_value = mock_config

    consumer = FraudTransactionConsumer()

    mock_consumer_cls.assert_called_once()
    assert consumer.repo == mock_repo.return_value

@patch("services.consumer.consumer.Consumer")
@patch("services.consumer.consumer.config_loader")
@patch("services.consumer.consumer.TransactionRepository")
def test_consumer_handle_message_success(mock_repo, mock_config_loader, mock_consumer_cls):
    mock_config_loader.load.return_value = MagicMock()
    consumer = FraudTransactionConsumer()
    
    msg_data = {
        "transaction_id": str(uuid4()),
        "event_time_seconds": 12345,
        "amount": 100.5,
        "features": {"f1": 0.5},
        "event_timestamp": "2023-01-01T12:00:00",
        "data_source": "test",
        "is_fraud": False,
        "created_at": "2023-01-01T12:00:00"
    }

    mock_msg = MagicMock()
    mock_msg.value.return_value = json.dumps(msg_data).encode('utf-8')

    consumer._handle_message(mock_msg)
    
    mock_repo.return_value.insert_if_not_exists.assert_called_once()
