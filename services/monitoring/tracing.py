"""Langfuse trace shape for one fraud investigation.

One trace == one `/ask` request. The LangChain `CallbackHandler` auto-creates a
generation span per agent turn and a span per tool call; this module adds the
root span, the trace-level attributes needed to find a trace again in production
(session, user, environment), and a retriever span carrying what the
auto-instrumentation cannot see (how many cases came back, how similar).

    TRACE  fraud_investigation             (agent)     prompt, top_k, session, user
      +- retrieval                         (retriever) n_returned, similarity stats
      +- agent_iteration_*                 (generation) auto-created, token usage
      +- tool spans                        (tool)      auto-created

The Langfuse client disables itself with a warning when LANGFUSE_* keys are
absent, and every call here is wrapped: a tracing outage must degrade to
untraced answers, never to failed answers.
"""

import logging
import os
from contextlib import ExitStack, contextmanager

from langfuse import Langfuse, propagate_attributes

from services.monitoring.masking import mask_langfuse_data, mask_langfuse_otel_spans

logger = logging.getLogger(__name__)

RETRIEVAL_MODE_VECTOR = "vector"
RETRIEVAL_MODE_KNOWN_FRAUD = "known_fraud"
RETRIEVAL_MODE_SUSPECTED = "suspected"
RETRIEVAL_MODE_BY_ID = "by_id"
RETRIEVAL_MODE_NONE = "none"


def resolve_environment() -> str:
    """Langfuse environment name, derived from APP_ENV.

    Without this every environment writes into the same bucket and error rates
    or cost figures mix dev experiments with production traffic. Langfuse
    requires lowercase alphanumerics, hyphens and underscores.
    """
    app_env = (os.getenv("APP_ENV") or "").strip().lower()
    return app_env if app_env else "local"


class InvestigationTrace:
    """Handle on the root span of one investigation."""

    def __init__(self, client, root_span):
        self._client = client
        self._root = root_span

    @property
    def trace_id(self) -> str | None:
        return getattr(self._root, "trace_id", None)

    @contextmanager
    def retrieval_span(self, tool_name: str):
        """Wraps the retriever step. Yields a callable taking the summary dict,
        because the metadata is only known once the rows come back."""
        try:
            span_context = self._client.start_as_current_observation(
                name="retrieval", as_type="retriever", input={"tool": tool_name}
            )
        except Exception:
            logger.error("Failed to open retrieval span", exc_info=True)
            yield lambda summary: None
            return

        with span_context as span:
            def record(summary: dict) -> None:
                try:
                    span.update(output=summary, metadata=summary)
                    if not summary.get("n_returned"):
                        span.update(level="WARNING", status_message="empty_retrieval")
                except Exception:
                    logger.error("Failed to record retrieval metadata", exc_info=True)

            yield record

    def finish(self, *, answer: str, trace_metadata: dict) -> None:
        try:
            self._root.update(output={"answer": answer}, metadata=trace_metadata)
        except Exception:
            logger.error("Failed to finalise investigation trace", exc_info=True)


class InvestigationTracer:
    """Owns the Langfuse client. Construct this before anything that calls
    `langfuse.get_client()` (notably `CallbackHandler`), since the client
    registered here — with masking and environment configured — is the one those
    callers will pick up.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str,
        service_name: str = "ms-fraud-rag",
        mask_sensitive_data: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.service_name = service_name
        self.environment = resolve_environment()
        # Sets the OTel `service.name` resource attribute (otherwise
        # "unknown_service"). Must happen before Langfuse() builds the tracer
        # provider — hence why this tracer is constructed before CallbackHandler.
        # An explicit env var (e.g. from the deploy) still wins over config.
        os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
        self.client = Langfuse(
            environment=self.environment,
            mask=mask_langfuse_data if mask_sensitive_data else None,
            # The LangChain handler's spans are third-party OTel spans, which
            # the `mask` hook above does not reach. They carry the raw payload,
            # so this second hook is the one that actually matters.
            mask_otel_spans=mask_langfuse_otel_spans if mask_sensitive_data else None,
        )
        logger.info(
            "Langfuse tracing initialised: environment=%s masking=%s",
            self.environment,
            mask_sensitive_data,
        )

    @contextmanager
    def investigation(
        self,
        *,
        prompt: str,
        top_k: int,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        tags = ["fraud-rag", f"provider:{self.provider}"]
        metadata = {"model": self.model, "provider": self.provider, "top_k": top_k}

        with ExitStack() as stack:
            try:
                root = stack.enter_context(
                    self.client.start_as_current_observation(
                        name="fraud_investigation",
                        as_type="agent",
                        input={"prompt": prompt, "top_k": top_k},
                        metadata=metadata,
                    )
                )
                stack.enter_context(
                    propagate_attributes(
                        session_id=session_id,
                        user_id=user_id,
                        tags=tags,
                        metadata=metadata,
                        # Otherwise the trace shows up unnamed in the Langfuse
                        # trace list.
                        trace_name=self.service_name,
                    )
                )
            except Exception:
                logger.error("Failed to open investigation trace; continuing untraced", exc_info=True)
                yield _NullInvestigationTrace()
                return

            yield InvestigationTrace(self.client, root)

    def flush(self) -> None:
        """Force-export pending spans. Needed by short-lived processes; the API
        server relies on the background exporter."""
        try:
            self.client.flush()
        except Exception:
            logger.error("Langfuse flush failed", exc_info=True)


class _NullInvestigationTrace:
    """Stand-in used when the trace could not be opened, so callers never need
    to null-check."""

    trace_id = None

    @contextmanager
    def retrieval_span(self, tool_name: str):
        yield lambda summary: None

    def finish(self, *, answer: str, trace_metadata: dict) -> None:
        return None
