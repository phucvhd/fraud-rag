import asyncio
import json
import logging
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

from schemas.dto import QueryRequest
from services.agent.state import GraphState
from services.agent.agent import LLMAgent
from services.monitoring.metrics import count_agent_iterations, sum_token_usage, summarize_retrieval
from services.monitoring.tracing import (
    RETRIEVAL_MODE_BY_ID,
    RETRIEVAL_MODE_KNOWN_FRAUD,
    RETRIEVAL_MODE_NONE,
    RETRIEVAL_MODE_SUSPECTED,
    RETRIEVAL_MODE_VECTOR,
    InvestigationTracer,
)
from shared.config_loader import config_loader

logger = logging.getLogger(__name__)

_CONTEXT_TOOL = "context_lookup"
_KNOWN_FRAUD_TOOL = "find_known_fraud"
_SUSPECTED_FRAUD_TOOL = "find_suspected_fraud"
_GET_TRANSACTION_TOOL = "get_transaction"
# Tools whose output is a transaction list, so auto_analyze enriches each row.
# fraud_stats is deliberately excluded: it returns aggregates, not transactions,
# and routes straight back to the agent to be narrated.
_LOOKUP_TOOLS = {_CONTEXT_TOOL, _KNOWN_FRAUD_TOOL, _SUSPECTED_FRAUD_TOOL, _GET_TRANSACTION_TOOL}

_RETRIEVAL_MODES = {
    _CONTEXT_TOOL: RETRIEVAL_MODE_VECTOR,
    _KNOWN_FRAUD_TOOL: RETRIEVAL_MODE_KNOWN_FRAUD,
    _SUSPECTED_FRAUD_TOOL: RETRIEVAL_MODE_SUSPECTED,
    _GET_TRANSACTION_TOOL: RETRIEVAL_MODE_BY_ID,
}

_AGENT_INSTRUCTIONS = """\
{prompt}

Instructions:
Choose the tool by what the user is really asking for:
 - a COUNT / RATE / TOTAL ("how many fraud today", "fraud rate this hour", "total
   amount flagged") -> fraud_stats. This returns aggregate numbers, not a list.
 - one SPECIFIC transaction by id ("show transaction <id>", "why was <id> flagged")
   -> get_transaction with that transaction_id.
 - "suspicious", "high risk", "most suspicious right now", "potential/possible fraud",
   "flagged", "not yet confirmed" -> find_suspected_fraud (transactions the model
   scored as risky, whether or not confirmed — this catches fraud BEFORE the label arrives).
 - "confirmed fraud", "known fraud cases", "already charged back" -> find_known_fraud
   (only transactions confirmed as fraudulent in the database).
 - a generic search by amount or free-text description -> context_lookup.
For the lookup tools (not fraud_stats/get_transaction) you MUST pass `top_k={top_k}` explicitly.
Translate the user's constraints into the tool's parameters rather than filtering afterwards:
 - an amount bound ("over 1000 EUR", "under 50") -> amount_min / amount_max
 - a time window ("in the last hour" -> since_hours=1; "today" -> since_days=1; "this week" -> since_days=7)
 - a minimum risk ("risk above 80%") -> min_risk=0.8
 - "highest risk" is already the default order of find_suspected_fraud; on find_known_fraud use order_by="risk"
 - "largest" / "biggest amount" -> order_by="amount" on find_known_fraud
If you called fraud_stats, summarise the returned numbers in plain language (totals, fraud rate,
etc.) instead of the per-transaction list below, then stop.
Otherwise, format your final response as a clear list of the transactions the tool returned.
For EACH transaction, clearly state:
 - Transaction ID
 - Transaction Time
 - Amount
 - Risk probability (the transaction's fraud_probability field, as a percentage; say "not available" if it is null)
 - Impact (your read of how risky it looks, based on the risk probability and the contributing features below,
   and whether it is already confirmed fraud via its is_fraud field)
 - Top contributing features (from the transaction's top_shap_features field — these are the specific
   features that drove THIS transaction's own score, already ranked by contribution; say "not available"
   if it is null. Do not just list raw V1/V2/etc. values — say which features pushed the score up or down.)
Do not filter out any results. Include all {top_k} transactions retrieved regardless of whether they are anomalous.
CRITICAL: After reviewing the data, you MUST generate a clear, human-readable text analysis. NEVER output raw JSON or strings like [TOOL_RESULT] or [END_TOOL_RESULT]."""


def _parse_lookup_payload(content) -> list | None:
    """Normalise a lookup tool's output into the list of transactions.

    The tool result `content` is either a plain JSON string or a list of content
    blocks (`[{"type": "text", "text": "..."}]`), depending on the model and
    LangChain version — some models return blocks even for text. The original
    `json.loads(content)` raised TypeError on the block form and silently
    skipped analysis, so both shapes are handled here. Returns None when the
    content is not a JSON array (e.g. "No data found.").
    """
    if isinstance(content, list):
        text = "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    elif isinstance(content, str):
        text = content
    else:
        text = ""

    if not text:
        return None

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    return parsed if isinstance(parsed, list) else None


@dataclass(frozen=True)
class InvestigationResult:
    answer: str
    # Returned to the caller so a user reporting a bad answer can be matched to
    # the trace that produced it. Without this, production reports are
    # unactionable: there is no way to find the request again.
    trace_id: str | None
    # The transactions the agent reasoned over. Not surfaced to /ask (QueryResponse
    # ignores it); the regression experiment needs it to score the answer against
    # the data the agent was actually given.
    retrieved: list = field(default_factory=list)


class FraudInspectorGraph:
    def __init__(self, agent: LLMAgent):
        cfg = config_loader.load()
        self.llm = agent.get_client()
        # Only the repository server is wired in. There was a second MCP server
        # exposing a per-transaction feature heuristic, but it recomputed a
        # static-correlation score from raw V-features (once per transaction,
        # over SSE) that duplicated — and sometimes contradicted — the real SHAP
        # attributions (`top_shap_features`) already in the retrieval payload.
        # The agent explains impact from that SHAP directly instead.
        self.mcp_client = MultiServerMCPClient({
            "repository_server": {
                "url": cfg.mcp_servers.repository.url,
                "transport": "sse",
            }
        })
        self.graph: CompiledStateGraph | None = None
        self._build_lock = asyncio.Lock()
        # Constructed before CallbackHandler on purpose: the tracer registers
        # the Langfuse client (with masking and environment set), and the
        # handler picks up whichever client is already registered.
        self._tracer = InvestigationTracer(
            model=cfg.llm.model_name,
            provider=cfg.llm.provider,
            service_name=cfg.monitoring.service_name,
            mask_sensitive_data=cfg.monitoring.mask_sensitive_data,
        )
        # Env-configured (LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL); no-ops if
        # unset. Gives one generation span per agent turn and one span per tool
        # call for free.
        self._langfuse_handler = CallbackHandler()

    async def build(self) -> CompiledStateGraph:
        async with self._build_lock:
            if self.graph is not None:
                return self.graph

            mcp_tools = await self.mcp_client.get_tools()
            llm_with_tools = self.llm.bind_tools(mcp_tools)

            async def agent_node(state: GraphState, config: RunnableConfig):
                messages = state["messages"]
                response = await llm_with_tools.ainvoke(messages, config)
                return {"messages": [response]}

            def capture_retrieval_node(state: GraphState):
                """Pull the retrieved transactions out of the last lookup result
                into state so run() can build the retriever span and the
                experiment can score against them. No per-transaction analysis
                call: the payload already carries fraud_probability and the real
                SHAP attributions, which the agent explains directly."""
                last_message = state["messages"][-1]
                transactions = _parse_lookup_payload(last_message.content)
                if transactions is None:
                    return {}

                # Last lookup wins if the agent retrieves twice.
                return {
                    "retrieved": transactions,
                    "retrieval_mode": _RETRIEVAL_MODES.get(
                        getattr(last_message, "name", None), RETRIEVAL_MODE_NONE
                    ),
                    "retrieval_tool": getattr(last_message, "name", None) or "unknown",
                }

            def route_after_tools(state: GraphState):
                last_message = state["messages"][-1]
                if getattr(last_message, "name", None) in _LOOKUP_TOOLS:
                    return "capture_retrieval"
                return "agent"

            workflow = StateGraph(GraphState)

            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", ToolNode(mcp_tools))
            workflow.add_node("capture_retrieval", capture_retrieval_node)

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", tools_condition)
            workflow.add_conditional_edges("tools", route_after_tools, {"capture_retrieval": "capture_retrieval", "agent": "agent"})
            workflow.add_edge("capture_retrieval", "agent")

            self.graph = workflow.compile()
            return self.graph

    async def run(self, request: QueryRequest) -> InvestigationResult:
        if not self.graph:
            await self.build()

        enriched_prompt = _AGENT_INSTRUCTIONS.format(prompt=request.prompt, top_k=request.top_k)
        initial_state = {"messages": [HumanMessage(content=enriched_prompt)]}

        with self._tracer.investigation(
            prompt=request.prompt,
            top_k=request.top_k,
            session_id=request.session_id,
            user_id=request.user_id,
        ) as trace:
            result = await self.graph.ainvoke(
                initial_state,
                config={"callbacks": [self._langfuse_handler]},
            )
            answer = result["messages"][-1].content
            self._record_trace_metadata(result, answer, trace)
            retrieved = [r for r in (result.get("retrieved") or []) if isinstance(r, dict)]
            return InvestigationResult(answer=answer, trace_id=trace.trace_id, retrieved=retrieved)

    def _record_trace_metadata(self, result: dict, answer: str, trace) -> None:
        """Retriever summary and trace-level counters.

        Wrapped whole: tracing is not allowed to turn a successful investigation
        into a failed request.
        """
        try:
            retrieved = [r for r in (result.get("retrieved") or []) if isinstance(r, dict)]
            retrieval_mode = result.get("retrieval_mode") or RETRIEVAL_MODE_NONE
            retrieval_tool = result.get("retrieval_tool") or "none"
            summary = summarize_retrieval(retrieved)

            # A metadata span rather than a timed one: real retrieval timings
            # come from the auto-instrumented MCP tool call, which cannot see
            # the rows it returned.
            with trace.retrieval_span(retrieval_tool) as record_retrieval:
                record_retrieval({**summary, "retrieval_mode": retrieval_mode})

            trace.finish(
                answer=answer,
                trace_metadata={
                    "retrieval_mode": retrieval_mode,
                    "n_returned": summary["n_returned"],
                    "n_missing_ml_score": summary["n_missing_ml_score"],
                    "n_iterations": count_agent_iterations(result["messages"]),
                    "token_usage": sum_token_usage(result["messages"]),
                },
            )
        except Exception:
            logger.error("Failed to record trace metadata for investigation", exc_info=True)
