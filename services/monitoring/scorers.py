"""Deterministic scorers for the agent's answer.

No LLM, no ground-truth labels: these check whether the answer *faithfully
reports the data the agent was given* — which is the agent's actual job here
(retrieve + present), and the kind of failure it actually has (dropping
transactions, misquoting a risk score, dumping raw JSON).

Deterministic on purpose: an LLM-as-judge would add cost and noise, and most of
what can go wrong here is structural, so a plain function catches it more
reliably. RAGAS-style semantic metrics can be layered on later for things these
cannot see (e.g. hallucinated *reasoning*).

Each scorer returns (value in [0,1], comment) or None when not applicable, so a
question with no scored transactions does not drag a metric down. `evaluate()`
assembles them into the shape the Langfuse experiment expects.
"""

import re

_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_RAW_MARKERS = ("[TOOL_RESULT]", "[END_TOOL_RESULT]")


def _retrieved_ids(retrieved: list[dict]) -> list[str]:
    return [
        str(r["transaction_id"])
        for r in retrieved
        if isinstance(r, dict) and r.get("transaction_id")
    ]


def transaction_coverage(answer: str, retrieved: list[dict]) -> tuple[float, str] | None:
    """Fraction of retrieved transactions whose id appears in the answer.

    The prompt requires all top_k transactions be listed; a model that silently
    drops some is the most common failure, and it is invisible unless checked.
    """
    ids = _retrieved_ids(retrieved)
    if not ids:
        return None
    mentioned = [tid for tid in ids if tid in answer]
    value = len(mentioned) / len(ids)
    missing = [tid for tid in ids if tid not in answer]
    comment = "all transactions listed" if not missing else f"missing {len(missing)}: {missing}"
    return value, comment


def no_hallucinated_ids(answer: str, retrieved: list[dict]) -> tuple[float, str] | None:
    """1.0 unless the answer cites transaction ids that were never retrieved."""
    ids = set(_retrieved_ids(retrieved))
    found = set(_UUID.findall(answer))
    if not found:
        # No ids cited at all — that is a coverage problem, not a hallucination.
        return 1.0, "no transaction ids in answer"
    invented = found - ids
    value = len(found & ids) / len(found)
    comment = "no invented ids" if not invented else f"invented {len(invented)}: {sorted(invented)}"
    return value, comment


def _percentage_candidates(score: float) -> set[str]:
    pct = score * 100
    return {f"{pct:.0f}", f"{pct:.1f}", f"{pct:.2f}", str(round(pct))}


def risk_score_fidelity(answer: str, retrieved: list[dict]) -> tuple[float, str] | None:
    """Fraction of scored transactions whose risk % is actually quoted.

    The prompt asks the answer to state each fraud_probability as a percentage;
    a model that hallucinates a different number is misreporting the ML score.
    Only transactions that *have* a score are counted — a null score is meant to
    be reported as "not available" and is out of scope here.
    """
    scored = [
        r for r in retrieved
        if isinstance(r, dict) and r.get("fraud_probability") is not None
    ]
    if not scored:
        return None
    matched = 0
    for r in scored:
        if _percentage_candidates(float(r["fraud_probability"])) & _substrings(answer):
            matched += 1
    value = matched / len(scored)
    comment = f"{matched}/{len(scored)} risk scores quoted"
    return value, comment


def _substrings(answer: str) -> set[str]:
    # Numbers the answer contains, so a percentage candidate can be matched
    # without a full parse. Cheap and good enough for a fidelity heuristic.
    return set(re.findall(r"\d+(?:\.\d+)?", answer))


def no_raw_output(answer: str, retrieved: list[dict]) -> tuple[float, str]:
    """0.0 if the answer leaked raw JSON or tool markers instead of prose."""
    stripped = (answer or "").strip()
    if any(marker in answer for marker in _RAW_MARKERS):
        return 0.0, "contains raw tool markers"
    if stripped.startswith(("[{", '{"', "[{'")):
        return 0.0, "looks like a raw JSON dump"
    return 1.0, "prose output"


_SCORERS = {
    "transaction_coverage": transaction_coverage,
    "no_hallucinated_ids": no_hallucinated_ids,
    "risk_score_fidelity": risk_score_fidelity,
    "no_raw_output": no_raw_output,
}


def evaluate(answer: str, retrieved: list[dict]) -> list[dict]:
    """Run every scorer, dropping the ones that do not apply.

    Returns dicts (not Langfuse objects) so this stays importable and testable
    without the SDK; the experiment runner wraps them into Evaluation.
    """
    results = []
    for name, scorer in _SCORERS.items():
        outcome = scorer(answer, retrieved)
        if outcome is None:
            continue
        value, comment = outcome
        results.append({"name": name, "value": value, "comment": comment, "data_type": "NUMERIC"})
    return results
