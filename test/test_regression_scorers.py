from services.monitoring.scorers import (
    evaluate,
    no_hallucinated_ids,
    no_raw_output,
    risk_score_fidelity,
    transaction_coverage,
)

A = "11111111-1111-1111-1111-111111111111"
B = "22222222-2222-2222-2222-222222222222"
C = "33333333-3333-3333-3333-333333333333"


def _txn(tid, score=None):
    return {"transaction_id": tid, "fraud_probability": score}


def test_coverage_full_when_all_ids_listed():
    answer = f"1. {A} ...\n2. {B} ..."
    value, _ = transaction_coverage(answer, [_txn(A), _txn(B)])
    assert value == 1.0


def test_coverage_partial_when_a_transaction_is_dropped():
    answer = f"Only one worth noting: {A}"
    value, comment = transaction_coverage(answer, [_txn(A), _txn(B)])
    assert value == 0.5
    assert B in comment


def test_coverage_not_applicable_without_retrieval():
    assert transaction_coverage("anything", []) is None


def test_hallucination_flags_invented_id():
    answer = f"{A} and also 99999999-9999-9999-9999-999999999999"
    value, comment = no_hallucinated_ids(answer, [_txn(A)])
    assert value == 0.5
    assert "invented" in comment


def test_hallucination_clean_when_all_ids_real():
    value, _ = no_hallucinated_ids(f"{A} {B}", [_txn(A), _txn(B)])
    assert value == 1.0


def test_hallucination_ok_when_no_ids_cited():
    # No ids at all is a coverage problem, not a hallucination.
    value, _ = no_hallucinated_ids("No suspicious activity found.", [_txn(A)])
    assert value == 1.0


def test_risk_fidelity_matches_quoted_percentage():
    # 0.73128 -> 73.1%
    answer = f"{A}: risk probability 73.1%"
    value, _ = risk_score_fidelity(answer, [_txn(A, 0.73128)])
    assert value == 1.0


def test_risk_fidelity_catches_misquoted_score():
    answer = f"{A}: risk probability 12%"
    value, _ = risk_score_fidelity(answer, [_txn(A, 0.73128)])
    assert value == 0.0


def test_risk_fidelity_not_applicable_without_scores():
    assert risk_score_fidelity("anything", [_txn(A, None)]) is None


def test_raw_output_flags_tool_markers():
    value, _ = no_raw_output("[TOOL_RESULT] ... [END_TOOL_RESULT]", [])
    assert value == 0.0


def test_raw_output_flags_json_dump():
    value, _ = no_raw_output('[{"transaction_id": "x"}]', [])
    assert value == 0.0


def test_raw_output_passes_prose():
    value, _ = no_raw_output("Here are the transactions you asked about.", [])
    assert value == 1.0


def test_evaluate_drops_not_applicable_scorers():
    # No retrieval and no scores: coverage + risk fidelity drop out, the two
    # answer-shape scorers remain.
    names = {r["name"] for r in evaluate("Some prose answer.", [])}
    assert names == {"no_hallucinated_ids", "no_raw_output"}


def test_evaluate_full_set_on_a_good_answer():
    answer = f"1. Transaction {A}, risk probability 73.1%. 2. Transaction {B}, risk 5.0%."
    results = {r["name"]: r["value"] for r in evaluate(answer, [_txn(A, 0.73128), _txn(B, 0.05)])}
    assert results["transaction_coverage"] == 1.0
    assert results["no_hallucinated_ids"] == 1.0
    assert results["risk_score_fidelity"] == 1.0
    assert results["no_raw_output"] == 1.0
