"""RAGAS semantic metrics — the layer the deterministic scorers cannot reach.

The deterministic scorers check structure (ids present, numbers match, filters
honoured). Two things they cannot see:

  * faithfulness  — is every claim in the answer grounded in the retrieved data,
    or did the model invent a supporting fact?
  * answer_relevancy — does the answer actually address the question, or wander?

Both use a judge, so both carry a judge-quality caveat — but not equally:
`faithfulness` leans entirely on the LLM (statement decomposition + entailment),
so a weak local judge returns noise. `answer_relevancy` leans on the *embeddings*
(the LLM only drafts candidate questions; scoring is cosine similarity), so with
a decent local embedding model it discriminates usefully even behind a weak LLM.

Everything is import-guarded and returns None on failure, so the experiment runs
fine without ragas installed or without a judge configured. Point `RAGAS_JUDGE_*`
at a strong model to make `faithfulness` trustworthy.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def _render_context(retrieved: list[dict]) -> list[str]:
    """One context string per retrieved transaction, as the judge sees it."""
    return [json.dumps(txn, default=str) for txn in retrieved if isinstance(txn, dict)]


class RagasEvaluator:
    """Lazily builds RAGAS faithfulness + answer_relevancy behind a shared judge
    LLM and embedding model. Every scoring call returns None on any failure so it
    can never break the experiment — a missing semantic score is acceptable, a
    failed run is not.
    """

    def __init__(self):
        self._built = False
        self._faithfulness = None
        self._answer_relevancy = None
        self._unavailable_reason: str | None = None

    def _build(self):
        if self._built:
            return
        self._built = True
        try:
            import instructor
            from openai import AsyncOpenAI
            from ragas.llms.base import InstructorLLM
            from ragas.metrics.collections import Faithfulness
        except ImportError as exc:
            self._unavailable_reason = f"ragas not installed: {exc}"
            return

        base_url = os.getenv("RAGAS_JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL") or "http://localhost:1234/v1"
        api_key = os.getenv("RAGAS_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"
        model = os.getenv("RAGAS_JUDGE_MODEL") or "google/gemma-4-e4b"
        emb_model = os.getenv("RAGAS_JUDGE_EMBEDDINGS") or "all-MiniLM-L6-v2"

        if "localhost" in base_url or "127.0.0.1" in base_url:
            logger.warning(
                "RAGAS judge LLM is local (%s): faithfulness will be NOISY (it is "
                "LLM-bound). answer_relevancy is embedding-bound and stays usable.",
                model,
            )
        try:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            # RAGAS hardcodes instructor Mode.JSON, which local OpenAI-compatible
            # servers reject; patch with JSON_SCHEMA ourselves (OpenAI accepts it too).
            patched = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
            judge = InstructorLLM(client=patched, model=model, provider="openai")
            self._faithfulness = Faithfulness(llm=judge)
        except Exception as exc:
            self._unavailable_reason = f"judge init failed: {exc}"
            logger.error("Could not build RAGAS judge LLM: %s", exc)
            return

        # answer_relevancy additionally needs embeddings; if they fail to load,
        # faithfulness still works, so build them separately.
        try:
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.metrics.collections import AnswerRelevancy

            embeddings = HuggingFaceEmbeddings(model=emb_model)
            self._answer_relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)
        except Exception as exc:
            logger.error("Could not build RAGAS answer_relevancy (embeddings): %s", exc)

    @property
    def unavailable_reason(self) -> str | None:
        self._build()
        return self._unavailable_reason

    async def faithfulness(self, question: str, answer: str, retrieved: list[dict]) -> tuple[float, str] | None:
        self._build()
        if self._faithfulness is None:
            return None
        contexts = _render_context(retrieved)
        if not contexts:
            return None  # undefined with no context to check against
        try:
            result = await self._faithfulness.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
            return float(result.value), (result.reason or "")[:500]
        except Exception as exc:
            logger.error("RAGAS faithfulness scoring failed: %s", exc, exc_info=True)
            return None

    async def answer_relevancy(self, question: str, answer: str) -> tuple[float, str] | None:
        self._build()
        if self._answer_relevancy is None or not (question and answer):
            return None
        try:
            result = await self._answer_relevancy.ascore(user_input=question, response=answer)
            return float(result.value), (result.reason or "")[:500]
        except Exception as exc:
            logger.error("RAGAS answer_relevancy scoring failed: %s", exc, exc_info=True)
            return None
