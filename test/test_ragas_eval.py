import asyncio
import json

from services.monitoring.ragas_eval import RagasEvaluator, _render_context

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


def test_faithfulness_none_without_context():
    ev = RagasEvaluator()
    ev._built = True  # skip real judge build
    ev._faithfulness = object()
    assert asyncio.run(ev.faithfulness("q", "an answer", [])) is None


def test_answer_relevancy_none_without_question_or_answer():
    ev = RagasEvaluator()
    ev._built = True
    ev._answer_relevancy = object()
    assert asyncio.run(ev.answer_relevancy("", "answer")) is None
    assert asyncio.run(ev.answer_relevancy("q", "")) is None


def test_scores_none_when_metrics_unavailable():
    ev = RagasEvaluator()
    ev._built = True  # nothing built -> both metrics None
    assert asyncio.run(ev.faithfulness("q", "a", [{"transaction_id": A}])) is None
    assert asyncio.run(ev.answer_relevancy("q", "a")) is None
