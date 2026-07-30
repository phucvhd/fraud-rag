import asyncio
import json
import logging
from datetime import datetime

from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, ToolMessage

from schemas.dto import QueryRequest
from services.agent.state import GraphState
from services.agent.agent import LLMAgent
from shared.config_loader import config_loader

logger = logging.getLogger(__name__)

_LOOKUP_TOOLS = {"context_lookup", "find_known_fraud", "query_transactions"}
_ANALYSIS_TOOL = "interpret_fraud_features"

_AGENT_INSTRUCTIONS = """\
{prompt}

Context:
The current time is {now} (ISO-8601, same clock the database timestamps use). Resolve every
relative time expression in the question ("in the last hour", "today", "since noon") against
this value and pass the result as start_time/end_time. Never guess the current date.

Instructions:
Pick ONE lookup tool:
- query_transactions — when the question has an EXACT filter: an amount threshold, a time window,
  or a fraud flag (e.g. "over 1000 EUR in the last hour"). Pass amount_min/amount_max, start_time/end_time
  (ISO-8601), and/or is_fraud. This is the accurate path for numeric/temporal questions.
- find_known_fraud — when the user asks broadly for recent anomalies/fraud with no specific filter;
  returns transactions confirmed fraudulent in the database.
- context_lookup — only for fuzzy, free-text descriptions with no numeric/time filter.
When invoking any lookup tool, you MUST explicitly pass `top_k={top_k}` rather than relying on its default.
A per-transaction fraud analysis (heuristic verdict and the real database label) is automatically
attached to your tool results — you do NOT need to call interpret_fraud_features yourself.
You MUST format your final response as a clear list containing all {top_k} transactions returned by the lookup tool.
For EACH transaction, clearly state:
 - Transaction ID
 - Transaction Time
 - Amount
 - Impact (use the analysis already provided to you)
 - Relevant Features (V1, V2, etc.)
Do not filter out any results. Include all {top_k} transactions retrieved regardless of whether they are anomalous.
CRITICAL: After reviewing the data, you MUST generate a clear, human-readable text analysis. NEVER output raw JSON or strings like [TOOL_RESULT] or [END_TOOL_RESULT]."""


class FraudInspectorGraph:
    def __init__(self, agent: LLMAgent):
        cfg = config_loader.load()
        self.llm = agent.get_client()
        self.mcp_client = MultiServerMCPClient({
            "analysis_server": {
                "url": cfg.mcp_servers.analysis.url,
                "transport": "sse",
            },
            "repository_server": {
                "url": cfg.mcp_servers.repository.url,
                "transport": "sse",
            }
        })
        self.graph: CompiledStateGraph | None = None
        self._build_lock = asyncio.Lock()

    async def build(self) -> CompiledStateGraph:
        async with self._build_lock:
            if self.graph is not None:
                return self.graph

            mcp_tools = await self.mcp_client.get_tools()
            llm_with_tools = self.llm.bind_tools(mcp_tools)
            analysis_tool = next(t for t in mcp_tools if t.name == _ANALYSIS_TOOL)

            async def agent_node(state: GraphState):
                messages = state["messages"]
                response = await llm_with_tools.ainvoke(messages)
                return {"messages": [response]}

            async def auto_analyze_node(state: GraphState):
                last_message = state["messages"][-1]
                try:
                    transactions = json.loads(last_message.content)
                except (json.JSONDecodeError, TypeError):
                    transactions = None

                if not isinstance(transactions, list):
                    return {"messages": []}

                lines = []
                for txn in transactions:
                    analysis = await analysis_tool.ainvoke({
                        "v_features": txn.get("features", {}),
                        "is_fraud": txn.get("is_fraud"),
                    })
                    lines.append(f"TransactionId: {txn.get('transaction_id')} -> {analysis}")

                combined = "\n".join(lines) if lines else "No transactions to analyze."
                return {
                    "messages": [
                        ToolMessage(content=combined, name=_ANALYSIS_TOOL, tool_call_id="auto-analyze")
                    ]
                }

            def route_after_tools(state: GraphState):
                last_message = state["messages"][-1]
                if getattr(last_message, "name", None) in _LOOKUP_TOOLS:
                    return "auto_analyze"
                return "agent"

            workflow = StateGraph(GraphState)

            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", ToolNode(mcp_tools))
            workflow.add_node("auto_analyze", auto_analyze_node)

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", tools_condition)
            workflow.add_conditional_edges("tools", route_after_tools, {"auto_analyze": "auto_analyze", "agent": "agent"})
            workflow.add_edge("auto_analyze", "agent")

            self.graph = workflow.compile()
            return self.graph

    async def run(self, request: QueryRequest) -> str:
        if not self.graph:
            await self.build()

        # Naive local time — matches the TIMESTAMP values the producer writes,
        # so relative windows the model derives line up with event_timestamp.
        enriched_prompt = _AGENT_INSTRUCTIONS.format(
            prompt=request.prompt,
            top_k=request.top_k,
            now=datetime.now().isoformat(timespec="seconds"),
        )
        initial_state = {"messages": [HumanMessage(content=enriched_prompt)]}

        result = await self.graph.ainvoke(initial_state)
        return result["messages"][-1].content
