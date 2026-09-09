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
def context_lookup(
    top_k: int,
    query: str | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
) -> str:
    """Search transactions by natural-language similarity, optionally narrowed to
    an amount range. Returns a JSON list of transactions with their amount, time,
    is_fraud label, fraud_probability (the model's risk score, 0-1, may be null),
    top_shap_features (may be null) and raw features.
    Pass `query` for a descriptive/semantic search (e.g. 'similar to card-testing').
    Pass amount_min / amount_max to restrict to an amount range. When the request
    is a PURE amount filter with no descriptive term ('transactions around 50 EUR'
    -> amount_min=40, amount_max=60), you may OMIT `query` entirely — do not invent
    one. 'over 1000 EUR' -> amount_min=1000, no query.
    Do NOT use this for anomaly/fraud/suspicious-transaction questions — use
    find_known_fraud instead, since this tool does not filter by the real fraud label.
    Always specify 'top_k' to define how many results to return."""
    try:
        logger.info("Start retrieving context")
        context = rag_engine.context_lookup(query, top_k, amount_min=amount_min, amount_max=amount_max)
        logger.info("Retrieved context successfully")
        return context
    except Exception as e:
        logger.error("Failed to retrieve context: %s", e)
        raise


@mcp.tool()
def find_known_fraud(
    top_k: int,
    amount_min: float | None = None,
    amount_max: float | None = None,
    min_risk: float | None = None,
    order_by: str = "recency",
) -> str:
    """Use this tool when the user asks for anomalies, fraud cases, or suspicious
    transactions. Returns transactions confirmed as fraudulent (is_fraud = true)
    directly from the database as a JSON list, each including fraud_probability
    (the model's risk score, 0-1, may be null) and top_shap_features (may be null).
    This is ground truth, not a similarity search.

    Honour the user's constraints via the parameters:
     - amount_min / amount_max: restrict to an amount range ('over 1000 EUR'
       -> amount_min=1000).
     - min_risk: only transactions with fraud_probability >= this (0-1).
     - order_by: 'recency' (default, newest first), 'risk' (highest
       fraud_probability first — use this for 'most suspicious' / 'highest risk'),
       or 'amount' (largest first).
    Always specify 'top_k' to define how many results to return."""
    try:
        logger.info("Start retrieving known fraud transactions")
        context = rag_engine.fraud_lookup(
            top_k, amount_min=amount_min, amount_max=amount_max, min_risk=min_risk, order_by=order_by
        )
        logger.info("Retrieved known fraud transactions successfully")
        return context
    except Exception as e:
        logger.error("Failed to retrieve known fraud transactions: %s", e)
        raise


if __name__ == "__main__":
    configure_logging()
    mcp.run(transport="sse")
