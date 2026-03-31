import asyncio

from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage

from schemas.dto import QueryRequest
from services.agent.state import GraphState
from services.agent.agent import LLMAgent
from shared.config_loader import config_loader


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

            async def agent_node(state: GraphState):
                messages = state["messages"]
                response = await llm_with_tools.ainvoke(messages)
                return {"messages": [response]}

            workflow = StateGraph(GraphState)

            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", ToolNode(mcp_tools))

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", tools_condition)
            workflow.add_edge("tools", "agent")

            self.graph = workflow.compile()
            return self.graph

    async def run(self, request: QueryRequest) -> str:
        if not self.graph:
            await self.build()

        enriched_prompt = (
            f"{request.prompt}\n\n"
            f"Instructions:\n"
            f"When invoking the context_lookup tool, you MUST explicitly pass `top_k={request.top_k}` as an argument rather than relying on its default value.\n"
            f"You MUST format your final response as a clear, list containing all {request.top_k} transactions.\n"
            f"For EACH transaction, clearly state:\n"
            f" - Transaction ID\n"
            f" - Transaction Time\n"
            f" - Amount\n"
            f" - Impact\n"
            f" - Relevant Features (V1, V2, etc.)\n"
            f"Do not filter out any results. Include all {request.top_k} transactions retrieved regardless of whether they are anomalous.\n"
            f"CRITICAL: If you use the interpret_fraud_features tool, you MUST pass a SINGLE dictionary containing the V-features for ONE transaction (e.g. {{'V1': 1.2, 'V2': -0.5}}). NEVER pass a list of dictionaries.\n"
            f"CRITICAL: After running your tools, you MUST analyze the data and generate a clear, human-readable text analysis. NEVER output raw strings like [TOOL_RESULT] or [END_TOOL_RESULT]."
        )
        initial_state = {"messages": [HumanMessage(content=enriched_prompt)]}

        result = await self.graph.ainvoke(initial_state)
        return result["messages"][-1].content
