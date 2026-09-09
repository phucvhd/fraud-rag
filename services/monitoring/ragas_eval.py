"""RAGAS faithfulness as an optional, semantic complement to the deterministic
scorers.

The deterministic scorers check that the answer's *structure* matches the data
(ids present, numbers quoted). They cannot catch a fluent claim the data does
not support — "this is fraud because the same card was charged back last week"
when no such fact was retrieved. RAGAS `faithfulness` uses an LLM judge to check
exactly that: is every statement in the answer grounded in the retrieved context.

Cost: it is LLM-as-judge, so it needs a *strong* judge to be meaningful. A weak
local judge (a 4B model) produces noisy scores — usable to prove the wiring, not
to gate a deploy. Point `RAGAS_JUDGE_*` at a strong model (e.g. gpt-4o) for real
use. This module is import-guarded so the experiment runs fine without ragas
installed or without a judge configured.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def _render_context(retrieved: list[dict]) -> list[str]:
    """One context string per retrieved transaction, as the judge sees it.

    Includes the same fields the agent was given so faithfulness is judged
    against the agent's actual evidence, not a cleaned-up version.
    """
    return [json.dumps(txn, default=str) for txn in retrieved if isinstance(txn, dict)]


class FaithfulnessJudge:
    """Lazily builds a RAGAS faithfulness metric backed by an OpenAI-compatible
    judge. Returns None from `score` on any failure so it can never break the
    experiment — a missing semantic score is acceptable, a failed run is not.
    """

    def __init__(self):
        self._metric = None
        self._unavailable_reason: str | None = None

    def _build(self):
        if self._metric is not None or self._unavailable_reason is not None:
            return
        try:
            import instructor
            from openai import AsyncOpenAI
            from ragas.llms.base import InstructorLLM
            from ragas.metrics.collections import Faithfulness
        except ImportError as exc:
            self._unavailable_reason = f"ragas not installed: {exc}"
            return

        # Judge config is separate from the agent's LLM: you want to judge with a
        # different (stronger) model than the one under test. Defaults fall back
        # to the local endpoint so it runs out of the box, with a loud warning.
        base_url = os.getenv("RAGAS_JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL") or "http://localhost:1234/v1"
        api_key = os.getenv("RAGAS_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"
        model = os.getenv("RAGAS_JUDGE_MODEL") or "google/gemma-4-e4b"

        if "localhost" in base_url or "127.0.0.1" in base_url:
            logger.warning(
                "RAGAS judge is a local model (%s); faithfulness scores will be NOISY. "
                "Set RAGAS_JUDGE_BASE_URL/MODEL/API_KEY to a strong model to gate on them.",
                model,
            )
        try:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            # RAGAS's llm_factory hardcodes instructor Mode.JSON, which local
            # OpenAI-compatible servers (e.g. LM Studio) reject — they want
            # json_schema. Patch the client ourselves with JSON_SCHEMA, which
            # both those servers and OpenAI accept, and hand InstructorLLM the
            # already-patched client (it does not re-patch).
            patched = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
            judge = InstructorLLM(client=patched, model=model, provider="openai")
            self._metric = Faithfulness(llm=judge)
        except Exception as exc:
            self._unavailable_reason = f"judge init failed: {exc}"
            logger.error("Could not build RAGAS faithfulness judge: %s", exc)

    @property
    def unavailable_reason(self) -> str | None:
        self._build()
        return self._unavailable_reason

    async def score(self, question: str, answer: str, retrieved: list[dict]) -> tuple[float, str] | None:
        self._build()
        if self._metric is None:
            return None
        contexts = _render_context(retrieved)
        if not contexts:
            # Faithfulness is undefined with no context to check against.
            return None
        try:
            result = await self._metric.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
            value = float(result.value)
            return value, (result.reason or "")[:500]
        except Exception as exc:
            logger.error("RAGAS faithfulness scoring failed: %s", exc, exc_info=True)
            return None
