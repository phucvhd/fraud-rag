import asyncio
import json
import logging

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

_LOOKUP_TOOLS = {"context_lookup", "find_known_fraud"}
_ANALYSIS_TOOL = "interpret_fraud_features"

_AGENT_INSTRUCTIONS = """\
{prompt}

Instructions:
Use the find_known_fraud tool (not context_lookup) when the user asks about anomalies, fraud, or
suspicious transactions — it returns transactions confirmed as fraudulent in the database.
Use context_lookup only for generic searches (e.g. by amount or free-text description).
When invoking either lookup tool, you MUST explicitly pass `top_k={top_k}` as an argument rather than relying on its default value.
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

        enriched_prompt = _AGENT_INSTRUCTIONS.format(prompt=request.prompt, top_k=request.top_k)
        initial_state = {"messages": [HumanMessage(content=enriched_prompt)]}

        result = await self.graph.ainvoke(initial_state)
        return result["messages"][-1].content
