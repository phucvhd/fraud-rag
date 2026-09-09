import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

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
def test_fraud_lookup_applies_amount_and_risk_filters(mock_config_loader, mock_get_engine):
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, [])

    engine.fraud_lookup(3, amount_min=1000, amount_max=5000, min_risk=0.8)

    sql = str(mock_conn.execute.call_args[0][0])
    assert "transactions.amount >=" in sql
    assert "transactions.amount <=" in sql
    assert "transactions.fraud_probability >=" in sql


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_fraud_lookup_order_by_risk_sorts_on_probability(mock_config_loader, mock_get_engine):
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, [])

    engine.fraud_lookup(3, order_by="risk")

    sql = str(mock_conn.execute.call_args[0][0])
    assert "ORDER BY transactions.fraud_probability DESC" in sql


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_fraud_lookup_unknown_order_falls_back_to_recency(mock_config_loader, mock_get_engine):
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, [])

    engine.fraud_lookup(3, order_by="'; DROP TABLE transactions; --")

    sql = str(mock_conn.execute.call_args[0][0])
    # A bad/injected order_by must degrade to recency, never reach the query.
    assert "ORDER BY transactions.event_timestamp DESC" in sql
    assert "DROP TABLE" not in sql


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_fraud_lookup_returns_serialized_json(mock_config_loader, mock_get_engine):
    records = [{
        "transaction_id": "7bc254fe-8d4b-433f-bfac-bc265b130eaa",
        "amount": Decimal("218.09"),
        "event_timestamp": "2026-03-27 15:30:26",
        "is_fraud": True,
        "fraud_probability": Decimal("0.92310"),
        "top_shap_features": {"V2": -0.9, "Time": -0.5},
        "features": {"V1": 4.4045},
    }]
    engine, _ = _build_engine(mock_get_engine, mock_config_loader, records)

    result = engine.fraud_lookup(3)
    payload = json.loads(result)

    assert payload[0]["is_fraud"] is True
    assert payload[0]["amount"] == 218.09
    assert payload[0]["fraud_probability"] == pytest.approx(0.9231)
    assert payload[0]["top_shap_features"] == {"V2": -0.9, "Time": -0.5}
    assert payload[0]["features"] == {"V1": 4.4045}
    # Not a vector search, so there is no similarity to report.
    assert payload[0]["similarity"] is None


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_context_lookup_reports_cosine_similarity(mock_config_loader, mock_get_engine):
    records = [{
        "transaction_id": "7bc254fe-8d4b-433f-bfac-bc265b130eaa",
        "amount": Decimal("218.09"),
        "event_timestamp": "2026-03-27 15:30:26",
        "is_fraud": False,
        "fraud_probability": Decimal("0.10000"),
        "top_shap_features": None,
        "features": {"V1": 4.4045},
        "cosine_distance": 0.25,
    }]
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, records)
    engine.embedder.encode.return_value.tolist.return_value = [0.1, 0.2]

    payload = json.loads(engine.context_lookup("anomalies", 3))

    assert payload[0]["similarity"] == pytest.approx(0.75)
    # Ranking must stay on l2 distance; cosine is selected for observability only.
    assert "<->" in str(mock_conn.execute.call_args[0][0])


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_fraud_lookup_serializes_null_probability(mock_config_loader, mock_get_engine):
    records = [{
        "transaction_id": "7bc254fe-8d4b-433f-bfac-bc265b130eaa",
        "amount": Decimal("218.09"),
        "event_timestamp": "2026-03-27 15:30:26",
        "is_fraud": True,
        "fraud_probability": None,
        "top_shap_features": None,
        "features": {"V1": 4.4045},
    }]
    engine, _ = _build_engine(mock_get_engine, mock_config_loader, records)

    result = engine.fraud_lookup(3)
    payload = json.loads(result)

    assert payload[0]["fraud_probability"] is None
    assert payload[0]["top_shap_features"] is None


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_context_lookup_no_data(mock_config_loader, mock_get_engine):
    engine, _ = _build_engine(mock_get_engine, mock_config_loader, [])
    engine.embedder.encode.return_value.tolist.return_value = [0.1, 0.2]

    assert engine.context_lookup("anomalies", 3) == "No data found."


@patch("services.tool.rag_tool.get_engine")
@patch("services.tool.rag_tool.config_loader")
def test_context_lookup_without_query_is_a_pure_amount_filter(mock_config_loader, mock_get_engine):
    records = [{
        "transaction_id": "7bc254fe-8d4b-433f-bfac-bc265b130eaa",
        "amount": Decimal("52.00"),
        "event_timestamp": "2026-03-27 15:30:26",
        "is_fraud": False,
        "fraud_probability": None,
        "top_shap_features": None,
        "features": {"V1": 1.0},
    }]
    engine, mock_conn = _build_engine(mock_get_engine, mock_config_loader, records)

    payload = json.loads(engine.context_lookup(top_k=3, amount_min=40, amount_max=60))

    # No descriptive term -> no embedding call, no vector search.
    engine.embedder.encode.assert_not_called()
    sql = str(mock_conn.execute.call_args[0][0])
    assert "transactions.amount >=" in sql and "transactions.amount <=" in sql
    assert "ORDER BY transactions.event_timestamp DESC" in sql
    assert payload[0]["similarity"] is None
