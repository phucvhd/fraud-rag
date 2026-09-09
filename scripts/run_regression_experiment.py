"""Run the fraud agent over the regression dataset and score every answer.

This is the automated regression: no human in the loop. Each question is sent
through the real agent, and the answer is scored by the deterministic checks in
`services/monitoring/scorers.py` (all transactions listed? no invented ids? risk
% matches the payload? prose, not raw JSON?). Scores land on a Langfuse
Experiment run, so two runs — e.g. before and after a prompt or model change —
sit side by side in the UI and a regression shows up as a dropped score.

Requires the full stack up (MCP servers, Postgres, the LLM endpoint), because it
exercises the real agent. Upload the dataset first with
`scripts/upload_regression_dataset.py`.

    LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... PYTHONPATH=. \
        python scripts/run_regression_experiment.py [run_name]

`run_name` defaults to a timestamp; pass something like "llama3-baseline" or
"gpt4o-newprompt" to label the comparison.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load LANGFUSE_* before get_client() / the graph read the environment.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from langfuse import Evaluation, get_client  # noqa: E402

from schemas.dto import QueryRequest
from services.agent.agent import LLMAgent
from services.agent.graph import FraudInspectorGraph
from services.monitoring import scorers
from services.monitoring.ragas_eval import FaithfulnessJudge
from shared.config_loader import config_loader
from shared.logging_config import configure_logging

FIXTURE = Path(__file__).resolve().parents[1] / "test/fixtures/regression_dataset.json"

# Built once, reused across items. build() is cached behind its own lock, so
# calling it per task is cheap after the first.
_inspector = FraudInspectorGraph(LLMAgent())


async def _task(*, item, **_kwargs):
    """Run the agent on one dataset item; return what the scorers need."""
    payload = item.input if hasattr(item, "input") else item["input"]
    await _inspector.build()
    result = await _inspector.run(QueryRequest(prompt=payload["prompt"], top_k=payload["top_k"]))
    return {
        "answer": result.answer,
        "retrieved": result.retrieved,
        "trace_id": result.trace_id,
    }


def _deterministic_evaluators(*, input, output, expected_output=None, metadata=None, **_kwargs):
    """Turn the pure scorers into Langfuse Evaluations for this item."""
    if not output:
        return [Evaluation(name="task_failed", value=0.0, data_type="NUMERIC", comment="no output")]
    return [
        Evaluation(
            name=score["name"],
            value=score["value"],
            comment=score["comment"],
            data_type=score["data_type"],
        )
        for score in scorers.evaluate(output.get("answer", ""), output.get("retrieved", []))
    ]


# Off by default: RAGAS is LLM-as-judge and only meaningful with a strong judge.
# With the local model it returns noise (empirically 0.0 for good and bad alike),
# so enable it only when RAGAS_JUDGE_* points at a capable model.
_ragas_judge = FaithfulnessJudge() if os.getenv("RAGAS_ENABLED") else None


async def _ragas_evaluators(*, input, output, expected_output=None, metadata=None, **_kwargs):
    """Semantic faithfulness — catches unsupported claims the structural scorers
    cannot see. Returns [] (no score) rather than failing when the judge is
    unavailable or the answer has no context to check against."""
    if _ragas_judge is None or not output:
        return []
    question = input.get("prompt", "") if isinstance(input, dict) else ""
    scored = await _ragas_judge.score(question, output.get("answer", ""), output.get("retrieved", []))
    if scored is None:
        return []
    value, reason = scored
    return [Evaluation(name="ragas_faithfulness", value=value, comment=reason, data_type="NUMERIC")]


def _load_data(client, dataset_name: str):
    """Prefer the uploaded Langfuse dataset (links the run to it in the UI);
    fall back to the local fixture so the experiment still runs if it was never
    uploaded."""
    try:
        dataset = client.get_dataset(dataset_name)
        if dataset.items:
            return dataset.items
    except Exception:
        pass
    spec = json.loads(FIXTURE.read_text())
    return [
        {"input": i["input"], "expected_output": i.get("expected_output"), "metadata": i.get("metadata")}
        for i in spec["items"]
    ]


def main() -> int:
    configure_logging()
    client = get_client()
    if not client.auth_check():
        print("Langfuse auth failed — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY.", file=sys.stderr)
        return 1

    spec = json.loads(FIXTURE.read_text())
    run_name = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    cfg = config_loader.load()

    result = client.run_experiment(
        name=spec["name"],
        run_name=run_name,
        description="Deterministic regression over the fixed fraud-agent question set.",
        data=_load_data(client, spec["name"]),
        task=_task,
        # Deterministic scorers always run (cheap, exact). RAGAS is appended only
        # when RAGAS_ENABLED is set and adds a semantic faithfulness score.
        evaluators=[_deterministic_evaluators] + ([_ragas_evaluators] if _ragas_judge else []),
        # The local model serves one request at a time; serial keeps the run
        # deterministic and avoids hammering the endpoint.
        max_concurrency=1,
        metadata={"model": cfg.llm.model_name, "provider": cfg.llm.provider},
    )
    client.flush()

    print(f"\nExperiment run '{run_name}' complete.\n")
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
