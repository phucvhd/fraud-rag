import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from typing_extensions import NotRequired, TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    # Written by auto_analyze so the monitoring layer in FraudInspectorGraph.run
    # can see the rows the agent actually reasoned over (their ML scores, their
    # retrieval similarity) without re-parsing the message history. Overwritten
    # rather than accumulated: if the agent retrieves twice, the last lookup is
    # the one its final answer is about.
    retrieved: NotRequired[list[dict]]
    retrieval_mode: NotRequired[str]
    retrieval_tool: NotRequired[str]
