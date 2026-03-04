from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.tools import tool

from schemas.dto import QueryRequest
from services.agent.agent import LLMAgent

class FraudInspector:
    rag_engine_ref = None

    def __init__(self, config, agent, rag_engine):
        self.cfg = config

        self.llm = LLMAgent()

        self.mcp_client = MultiServerMCPClient({
            "pca_interpreter": {
                "url": "http://localhost:8003/sse",
                "transport": "sse",
            }
        })
        self.agent_executor = None
        FraudInspector.rag_engine_ref = rag_engine

    @tool
    @staticmethod
    def historical_db_lookup(query: str, top_k: int = 5):
        """
        Search for past fraud cases in PostgreSQL.
        Will automatically use the top_k specified in the initial request.
        """
        return FraudInspector.rag_engine_ref.historical_db_lookup(query, top_k)

    async def build_agent(self):
        mcp_tools = await self.mcp_client.get_tools()

        all_tools = mcp_tools + [self.historical_db_lookup]
        client = self.llm.get_client()
        self.agent_executor = create_agent(client, all_tools)

    async def run(self, request: QueryRequest):
        try:
            if not self.agent_executor:
                await self.build_agent()

            enriched_prompt = f"{request.prompt} (Note: retrieve top {request.top_k} relevant cases)"
            inputs = {"messages": [HumanMessage(content=enriched_prompt)]}
            result = await self.agent_executor.ainvoke(inputs)
            return result["messages"][-1].content
        except Exception as e:
            print(f"--- AGENT ERROR LOG ---")
            import traceback
            traceback.print_exc()
            print(f"-----------------------")
            raise e