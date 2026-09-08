"""Trace metadata derived from a completed investigation.

Pure functions, no I/O: what the auto-instrumented Langfuse spans cannot know
(how many cases came back, how similar they were, how many turns the agent took)
computed from the graph's final state.
"""

from statistics import mean


def summarize_retrieval(records: list[dict]) -> dict:
    """Metadata for the retriever span.

    `similarity` is None for the ground-truth lookup path, which is a SQL filter
    rather than a vector search — reporting 0.0 there would look like a
    catastrophic retrieval failure on every such trace.
    """
    similarities = [
        float(r["similarity"])
        for r in records
        if isinstance(r, dict) and r.get("similarity") is not None
    ]
    n_missing_ml_score = sum(
        1 for r in records if isinstance(r, dict) and r.get("fraud_probability") is None
    )

    return {
        "n_returned": len(records),
        # A transaction reaching the agent unscored means the answer cannot
        # report a risk probability for it — worth seeing on the trace.
        "n_missing_ml_score": n_missing_ml_score,
        "similarity": (
            {
                "mean": mean(similarities),
                "min": min(similarities),
                "max": max(similarities),
            }
            if similarities
            else None
        ),
        "transaction_ids": [
            str(r.get("transaction_id")) for r in records if isinstance(r, dict)
        ],
    }


def count_agent_iterations(messages: list) -> int:
    """How many times the agent LLM spoke in this trace.

    A healthy investigation is 2-3 turns; a tool-thrashing one climbs, which is
    the cheapest available stuck-detector. Counted from message types rather
    than a manual tally so it stays correct if the graph topology changes.
    """
    return sum(1 for m in messages if getattr(m, "type", None) == "ai")


def sum_token_usage(messages: list) -> dict[str, int] | None:
    """Aggregate token usage across every LLM turn in the trace.

    Returns None when no turn reported usage — true for OpenAI-compatible local
    servers that omit the field, and we would rather log "unknown" than a
    confident 0 that pollutes cost dashboards.
    """
    totals = {"input": 0, "output": 0, "total": 0}
    seen = False

    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        seen = True
        totals["input"] += usage.get("input_tokens", 0) or 0
        totals["output"] += usage.get("output_tokens", 0) or 0
        totals["total"] += usage.get("total_tokens", 0) or 0

    return totals if seen else None
