from types import SimpleNamespace

from services.monitoring.metrics import (
    count_agent_iterations,
    sum_token_usage,
    summarize_retrieval,
)

TXN = "11111111-1111-1111-1111-111111111111"


def test_summarize_retrieval_reports_similarity_stats():
    summary = summarize_retrieval([
        {"transaction_id": TXN, "similarity": 0.8, "fraud_probability": 0.4},
        {"transaction_id": "b", "similarity": 0.4, "fraud_probability": None},
    ])
    assert summary["n_returned"] == 2
    assert summary["n_missing_ml_score"] == 1
    assert summary["similarity"] == {"mean": 0.6000000000000001, "min": 0.4, "max": 0.8}
    assert summary["transaction_ids"] == [TXN, "b"]


def test_summarize_retrieval_leaves_similarity_null_for_non_vector_lookup():
    summary = summarize_retrieval([{"transaction_id": TXN, "similarity": None, "fraud_probability": 0.9}])
    assert summary["similarity"] is None


def test_summarize_retrieval_handles_no_results():
    summary = summarize_retrieval([])
    assert summary == {
        "n_returned": 0,
        "n_missing_ml_score": 0,
        "similarity": None,
        "transaction_ids": [],
    }


def test_count_agent_iterations_counts_only_llm_turns():
    messages = [
        SimpleNamespace(type="human"),
        SimpleNamespace(type="ai"),
        SimpleNamespace(type="tool"),
        SimpleNamespace(type="ai"),
    ]
    assert count_agent_iterations(messages) == 2


def test_sum_token_usage_aggregates_across_turns():
    messages = [
        SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
        SimpleNamespace(usage_metadata={"input_tokens": 20, "output_tokens": 1, "total_tokens": 21}),
    ]
    assert sum_token_usage(messages) == {"input": 30, "output": 6, "total": 36}


def test_sum_token_usage_returns_none_when_no_turn_reported_usage():
    assert sum_token_usage([SimpleNamespace(usage_metadata=None)]) is None
