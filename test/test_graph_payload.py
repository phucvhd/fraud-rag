import json

from services.agent.graph import _parse_lookup_payload

_TXNS = [{"transaction_id": "a", "features": {"V1": 1.0}, "is_fraud": True}]


def test_parses_a_plain_json_string():
    assert _parse_lookup_payload(json.dumps(_TXNS)) == _TXNS


def test_parses_a_list_of_content_blocks():
    # The shape returned by some models/LangChain versions — this is exactly
    # what silently broke auto_analyze and raised the retrieval WARNING.
    content = [{"type": "text", "text": json.dumps(_TXNS)}]
    assert _parse_lookup_payload(content) == _TXNS


def test_concatenates_multiple_text_blocks():
    payload = json.dumps(_TXNS)
    content = [
        {"type": "text", "text": payload[:10]},
        {"type": "text", "text": payload[10:]},
    ]
    assert _parse_lookup_payload(content) == _TXNS


def test_no_data_string_is_not_a_list():
    assert _parse_lookup_payload("No data found.") is None


def test_json_object_rather_than_array_is_rejected():
    assert _parse_lookup_payload(json.dumps({"transaction_id": "a"})) is None


def test_malformed_json_returns_none():
    assert _parse_lookup_payload('[{"transaction_id":') is None


def test_empty_and_non_text_content_returns_none():
    assert _parse_lookup_payload([]) is None
    assert _parse_lookup_payload([{"type": "image", "url": "x"}]) is None
    assert _parse_lookup_payload(None) is None
