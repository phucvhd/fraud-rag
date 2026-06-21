import logging

from mcp.server.fastmcp import FastMCP

from services.agent.sentence_transformer import SentenceTransformerModel
from services.tool.rag_tool import RAGQueryEngine
from shared.config_loader import config_loader
from shared.logging_config import configure_logging

logger = logging.getLogger(__name__)
mcp = FastMCP("Repository", port=8003)

cfg = config_loader.load()
sentence_transformer_model = SentenceTransformerModel()
rag_engine = RAGQueryEngine(sentence_transformer_model)


@mcp.tool()
def context_lookup(query: str, top_k: int) -> str:
    """Semantic search for transactions by natural language similarity
    (e.g., 'find transactions over 1000 EUR'). Returns a JSON list of transactions
    with their amount, time, is_fraud label and raw features.
    Do NOT use this for anomaly/fraud/suspicious-transaction questions — use
    find_known_fraud instead, since this tool does not filter by the real fraud label.
    Always specify 'top_k' to define how many results to return."""
    try:
        logger.info("Start retrieving context")
        context = rag_engine.context_lookup(query, top_k)
        logger.info("Retrieved context successfully")
        return context
    except Exception as e:
        logger.error("Failed to retrieve context: %s", e)
        raise


@mcp.tool()
def find_known_fraud(top_k: int) -> str:
    """Use this tool when the user asks for anomalies, fraud cases, or suspicious
    transactions. Returns the most recent transactions confirmed as fraudulent
    (is_fraud = true) directly from the database as a JSON list, ordered by recency.
    This is ground truth, not a similarity search.
    Always specify 'top_k' to define how many results to return."""
    try:
        logger.info("Start retrieving known fraud transactions")
        context = rag_engine.fraud_lookup(top_k)
        logger.info("Retrieved known fraud transactions successfully")
        return context
    except Exception as e:
        logger.error("Failed to retrieve known fraud transactions: %s", e)
        raise


if __name__ == "__main__":
    configure_logging()
    mcp.run(transport="sse")
