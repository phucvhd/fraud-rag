import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

from services.tool.rag_tool import RAGQueryEngine


def _build_engine(mock_get_engine, mock_config_loader, records):
    mock_config_loader.load.return_value = MagicMock()
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_conn.execute.return_value.mappings.return_value.all.return_value = records
    return RAGQueryEngine(MagicMock()), mock_conn


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_fraud_lookup_filters_by_is_fraud(mock_config_loader, mock_get_engine):
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, [])

    engine.fraud_lookup(3)

    stmt = mock_conn.execute.call_args[0][0]
    assert "transactions.is_fraud" in str(stmt)
    assert "ORDER BY" in str(stmt)


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_fraud_lookup_returns_serialized_json(mock_config_loader, mock_get_engine):
    records = [{
        "transaction_id": "7bc254fe-8d4b-433f-bfac-bc265b130eaa",
        "amount": Decimal("218.09"),
        "event_timestamp": "2026-03-27 15:30:26",
        "is_fraud": True,
        "features": {"V1": 4.4045},
    }]
    engine, _ = _build_engine(mock_get_engine, mock_config_loader, records)

    result = engine.fraud_lookup(3)
    payload = json.loads(result)

    assert payload[0]["is_fraud"] is True
    assert payload[0]["amount"] == 218.09
    assert payload[0]["features"] == {"V1": 4.4045}


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_context_lookup_no_data(mock_config_loader, mock_get_engine):
    engine, _ = _build_engine(mock_get_engine, mock_config_loader, [])
    engine.embedder.encode.return_value.tolist.return_value = [0.1, 0.2]

    assert engine.context_lookup("anomalies", 3) == "No data found."
