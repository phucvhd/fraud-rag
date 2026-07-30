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


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_query_transactions_applies_structured_filters(mock_config_loader, mock_get_engine):
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, [])

    engine.query_transactions(
        top_k=5,
        amount_min=1000,
        start_time="2026-07-29T13:00:00",
        end_time="2026-07-29T14:00:00",
        is_fraud=True,
    )

    stmt = str(mock_conn.execute.call_args[0][0])
    assert "transactions.amount >=" in stmt
    assert "transactions.event_timestamp >=" in stmt
    assert "transactions.event_timestamp <" in stmt
    assert "transactions.is_fraud" in stmt
    # No query -> most recent first, no embedding join.
    assert "ORDER BY transactions.event_timestamp DESC" in stmt
    assert "transaction_embeddings" not in stmt


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_query_transactions_hybrid_ranks_by_similarity(mock_config_loader, mock_get_engine):
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, [])
    engine.embedder.encode.return_value.tolist.return_value = [0.1, 0.2]

    engine.query_transactions(top_k=3, amount_min=1000, query="wire transfers")

    engine.embedder.encode.assert_called_once_with("wire transfers")
    stmt = str(mock_conn.execute.call_args[0][0])
    # With a query the filtered rows are joined to embeddings and ranked by distance.
    assert "transaction_embeddings" in stmt
    assert "transactions.amount >=" in stmt


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_query_transactions_invalid_time_returns_message(mock_config_loader, mock_get_engine):
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, [])

    result = engine.query_transactions(top_k=5, start_time="last tuesday")

    assert "Invalid start_time" in result
    mock_conn.execute.assert_not_called()


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_query_transactions_clamps_top_k(mock_config_loader, mock_get_engine):
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, [])

    engine.query_transactions(top_k=999)

    stmt = mock_conn.execute.call_args[0][0]
    assert 50 in stmt.compile().params.values()
