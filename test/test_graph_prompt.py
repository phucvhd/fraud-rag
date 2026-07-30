"""Guards the agent prompt contract.

The prompt is built with str.format, so a placeholder added to the template
without a matching argument in run() raises KeyError at request time. These
tests fail fast instead.
"""

import re
from datetime import datetime

from services.agent.graph import _AGENT_INSTRUCTIONS, _LOOKUP_TOOLS

EXPECTED_PLACEHOLDERS = {"prompt", "top_k", "now"}


def test_template_placeholders_are_exactly_the_supported_set():
    found = set(re.findall(r"\{(\w+)\}", _AGENT_INSTRUCTIONS))
    assert found == EXPECTED_PLACEHOLDERS


def test_template_formats_with_all_arguments():
    rendered = _AGENT_INSTRUCTIONS.format(
        prompt="Any anomaly over 1000 EUR in the last hour?",
        top_k=5,
        now=datetime(2026, 7, 29, 21, 0, 0).isoformat(timespec="seconds"),
    )

    assert "{" not in rendered.replace("{}", "")  # nothing left unsubstituted
    assert "2026-07-29T21:00:00" in rendered
    assert "Any anomaly over 1000 EUR in the last hour?" in rendered


def test_prompt_steers_relative_time_resolution():
    # The structured-filter path depends on the model resolving "last hour"
    # against a supplied clock rather than guessing.
    assert "current time" in _AGENT_INSTRUCTIONS.lower()
    assert "start_time" in _AGENT_INSTRUCTIONS


def test_structured_query_tool_results_get_auto_analysed():
    assert "query_transactions" in _LOOKUP_TOOLS
