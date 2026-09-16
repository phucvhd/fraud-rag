import json

from langfuse import (
    LangfuseOtelSpanAttributes,
    MaskOtelSpansParams,
    OtelSpanData,
    OtelSpanIdentifier,
)

from services.monitoring.masking import (
    REDACTED,
    mask_langfuse_data,
    mask_langfuse_otel_spans,
    redact,
)


def test_redacts_amount_and_features_at_any_depth():
    masked = redact({"input": {"transaction_id": "abc", "amount": 218.09, "features": {"V1": 4.4}}})
    assert masked["input"]["transaction_id"] == "abc"
    assert masked["input"]["amount"] == REDACTED
    assert masked["input"]["features"] == REDACTED


def test_redacts_inside_a_json_string():
    # Tool results reach the span as a JSON *string*; without parsing it the
    # whole retrieval payload would be exported untouched.
    payload = json.dumps([{"transaction_id": "abc", "amount": 10.0, "features": {"V1": 1.0}}])
    masked = redact(payload)
    assert json.loads(masked)[0]["amount"] == REDACTED
    assert json.loads(masked)[0]["transaction_id"] == "abc"


def test_leaves_free_text_alone():
    text = "Transaction abc has an amount of 218.09 EUR"
    assert redact(text) == text


def test_keeps_non_sensitive_fields_intact():
    masked = redact({"fraud_probability": 0.93, "similarity": 0.8, "n_returned": 5})
    assert masked == {"fraud_probability": 0.93, "similarity": 0.8, "n_returned": 5}


def test_mask_hook_redacts_everything_if_it_fails():
    class Exploding:
        def __iter__(self):
            raise RuntimeError("boom")

    # Failing open would ship unredacted data.
    assert mask_langfuse_data(data={"amount": Exploding()}) is not None


def _span(span_id: str, attributes: dict):
    return OtelSpanIdentifier(trace_id="t", span_id=span_id), OtelSpanData(
        trace_id="t",
        span_id=span_id,
        parent_span_id=None,
        name="tool",
        instrumentation_scope_name="langchain",
        instrumentation_scope_version=None,
        attributes=attributes,
        resource_attributes={},
    )


def test_otel_hook_patches_only_spans_that_changed():
    key = LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT
    sensitive_id, sensitive_span = _span("s1", {key: json.dumps({"amount": 10.0})})
    plain_id, plain_span = _span("s2", {key: "plain prose"})
    params = MaskOtelSpansParams(spans={sensitive_id: sensitive_span, plain_id: plain_span})

    result = mask_langfuse_otel_spans(params=params)

    assert set(result.span_patches) == {sensitive_id}
    patched = result.span_patches[sensitive_id].set_attributes
    assert json.loads(patched[key])["amount"] == REDACTED


def test_otel_hook_returns_none_when_nothing_needed_masking():
    identifier, span = _span("s1", {"some.other.key": "value"})
    params = MaskOtelSpansParams(spans={identifier: span})
    assert mask_langfuse_otel_spans(params=params) is None


def test_otel_hook_never_raises():
    # A raised exception here makes Langfuse drop the entire export batch.
    assert mask_langfuse_otel_spans(params=None) is None
