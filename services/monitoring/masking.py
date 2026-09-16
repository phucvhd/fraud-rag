"""Redacts transaction data before it leaves the process for Langfuse.

Traces otherwise carry the full retrieval payload — amounts and all 28 V-features
per transaction — to a third-party service. That is a data-protection question
rather than a technical one, so redaction happens here, at the boundary.

Two hooks are needed because traces come from two places:

  * `mask_langfuse_data` covers spans this codebase creates through the SDK.
  * `mask_langfuse_otel_spans` covers spans created by the LangChain callback
    handler, which the SDK-level `mask` does not see. This is the bigger
    surface: the tool-call spans contain the raw JSON payload.

Known limitation: only *structured* fields are redacted. The agent's final prose
answer quotes amounts in free text and cannot be masked reliably, so it still
reaches Langfuse. Self-host if that is unacceptable.
"""

import json
import logging
from typing import Any

from langfuse import (
    LangfuseOtelSpanAttributes,
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanPatch,
)

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

# Field names redacted wherever they appear, at any nesting depth.
SENSITIVE_KEYS = frozenset({"amount", "features", "top_shap_features", "v_features"})

_MASKED_ATTRIBUTES = (
    LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
    LangfuseOtelSpanAttributes.TRACE_INPUT,
    LangfuseOtelSpanAttributes.TRACE_OUTPUT,
)

_MAX_DEPTH = 10


def redact(data: Any, _depth: int = 0) -> Any:
    """Recursively redact sensitive keys.

    Strings are inspected too: tool results arrive as a JSON *string* nested
    inside the span payload, so without parsing them the payload would pass
    through untouched.
    """
    if _depth > _MAX_DEPTH:
        return data
    if isinstance(data, dict):
        return {
            key: (REDACTED if key in SENSITIVE_KEYS else redact(value, _depth + 1))
            for key, value in data.items()
        }
    if isinstance(data, (list, tuple)):
        return [redact(item, _depth + 1) for item in data]
    if isinstance(data, str):
        return _redact_embedded_json(data, _depth)
    return data


def _redact_embedded_json(text: str, depth: int) -> str:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        # Free text (including the agent's final answer) is left alone: there is
        # no reliable way to strip amounts out of prose.
        return text
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return text

    masked = redact(parsed, depth + 1)
    return text if masked == parsed else json.dumps(masked)


def mask_langfuse_data(*, data: Any, **_kwargs: Any) -> Any:
    """`mask` hook for the Langfuse SDK."""
    try:
        return redact(data)
    except Exception:
        # Failing open would ship unredacted data; redact everything instead.
        logger.error("Masking failed; redacting the whole payload", exc_info=True)
        return REDACTED


def mask_langfuse_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
    """`mask_otel_spans` hook, applied at export to third-party OTel spans.

    Returns None when nothing needed changing so Langfuse can skip the batch
    rewrite. A raised exception here would make Langfuse drop the whole export
    batch, so everything is caught.
    """
    try:
        patches: dict = {}
        for identifier, span in params.spans.items():
            replacements = {}
            for key in _MASKED_ATTRIBUTES:
                value = span.attributes.get(key)
                if not isinstance(value, str):
                    continue
                masked = redact(value)
                if masked != value:
                    replacements[key] = masked
            if replacements:
                patches[identifier] = OtelSpanPatch(set_attributes=replacements)
        return MaskOtelSpansResult(span_patches=patches) if patches else None
    except Exception:
        logger.error("OTel span masking failed; leaving batch unchanged", exc_info=True)
        return None
