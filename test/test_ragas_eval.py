import json

from services.monitoring.ragas_eval import FaithfulnessJudge, _render_context

A = "11111111-1111-1111-1111-111111111111"


def test_render_context_one_string_per_transaction():
    contexts = _render_context([
        {"transaction_id": A, "amount": 98.0},
        {"transaction_id": "b", "amount": 5.0},
    ])
    assert len(contexts) == 2
    assert json.loads(contexts[0])["transaction_id"] == A


def test_render_context_skips_non_dicts():
    assert _render_context([{"transaction_id": A}, "junk", None]) == [json.dumps({"transaction_id": A})]


def test_score_returns_none_without_context(monkeypatch):
    judge = FaithfulnessJudge()
    # Force the metric to look built so we exercise the empty-context guard
    # rather than trying to reach a live judge.
    judge._metric = object()

    import asyncio

    result = asyncio.run(judge.score("q", "an answer", []))
    assert result is None
