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
    since_hours: float | None = None,
    since_days: float | None = None,
) -> str:
    """Search transactions by natural-language similarity, optionally narrowed to
    an amount range and/or a recent time window. Returns a JSON list of transactions
    with their amount, time, is_fraud label, fraud_probability (0-1, may be null),
    top_shap_features (may be null) and raw features.
    Pass `query` for a descriptive/semantic search (e.g. 'similar to card-testing').
    Pass amount_min / amount_max to restrict to an amount range. When the request
    is a PURE amount filter with no descriptive term ('transactions around 50 EUR'
    -> amount_min=40, amount_max=60), you may OMIT `query` entirely — do not invent
    one. 'over 1000 EUR' -> amount_min=1000, no query.
    Pass since_hours / since_days for a recent window ('in the last hour' ->
    since_hours=1; 'today' -> since_days=1; 'this week' -> since_days=7).
    Do NOT use this for anomaly/fraud/suspicious-transaction questions — use
    find_known_fraud instead, since this tool does not filter by the real fraud label.
    Always specify 'top_k' to define how many results to return."""
    try:
        logger.info("Start retrieving context")
        context = rag_engine.context_lookup(
            query, top_k, amount_min=amount_min, amount_max=amount_max,
            since_hours=since_hours, since_days=since_days,
        )
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
    since_hours: float | None = None,
    since_days: float | None = None,
) -> str:
    """Use this ONLY for transactions already CONFIRMED as fraud (is_fraud = true)
    — e.g. 'known fraud cases', 'past confirmed fraud', 'transactions that were
    charged back'. Returns them directly from the database as a JSON list, each
    including fraud_probability and top_shap_features.
    For transactions that are merely SUSPECTED / high-risk but not yet confirmed,
    use find_suspected_fraud instead — this tool cannot see them.

    Honour the user's constraints via the parameters:
     - amount_min / amount_max: restrict to an amount range ('over 1000 EUR'
       -> amount_min=1000).
     - min_risk: only transactions with fraud_probability >= this (0-1).
     - order_by: 'recency' (default, newest first), 'risk' (highest
       fraud_probability first) or 'amount' (largest first).
     - since_hours / since_days: recent window ('in the last hour' -> since_hours=1;
       'today' -> since_days=1).
    Always specify 'top_k' to define how many results to return."""
    try:
        logger.info("Start retrieving known fraud transactions")
        context = rag_engine.fraud_lookup(
            top_k, amount_min=amount_min, amount_max=amount_max, min_risk=min_risk, order_by=order_by,
            since_hours=since_hours, since_days=since_days,
        )
        logger.info("Retrieved known fraud transactions successfully")
        return context
    except Exception as e:
        logger.error("Failed to retrieve known fraud transactions: %s", e)
        raise


@mcp.tool()
def find_suspected_fraud(
    top_k: int,
    min_risk: float | None = None,
    amount_min: float | None = None,
    amount_max: float | None = None,
    since_hours: float | None = None,
    since_days: float | None = None,
) -> str:
    """Use this for SUSPICIOUS / high-risk / 'most suspicious right now' / 'highest
    fraud risk' / 'potential fraud' questions — transactions the ML model scored as
    risky, whether or not they are confirmed fraud yet. Returns them ordered by
    fraud_probability (highest risk first) as a JSON list; each includes is_fraud
    so you can see whether it is already confirmed or still only suspected.

    This is the tool for catching fraud BEFORE the chargeback/analyst label
    arrives. Unlike find_known_fraud it does NOT require is_fraud = true, so it
    surfaces high-risk transactions that are not yet confirmed.

     - min_risk: only transactions with fraud_probability >= this (0-1), e.g.
       'risk above 80%' -> min_risk=0.8.
     - amount_min / amount_max: restrict to an amount range.
     - since_hours / since_days: recent window ('in the last hour' -> since_hours=1).
    Always specify 'top_k'."""
    try:
        logger.info("Start retrieving suspected fraud transactions")
        context = rag_engine.suspected_lookup(
            top_k, min_risk=min_risk, amount_min=amount_min, amount_max=amount_max,
            since_hours=since_hours, since_days=since_days,
        )
        logger.info("Retrieved suspected fraud transactions successfully")
        return context
    except Exception as e:
        logger.error("Failed to retrieve suspected fraud transactions: %s", e)
        raise


@mcp.tool()
def get_transaction(transaction_id: str) -> str:
    """Fetch the full record for ONE specific transaction by its id (a UUID).
    Use this when the user references a particular transaction — 'show me
    transaction <id>', 'details of <id>', 'why was <id> flagged'. Returns a JSON
    list with that single transaction (amount, time, is_fraud, fraud_probability,
    top_shap_features, features), or 'No data found.' if the id does not exist."""
    try:
        logger.info("Fetching transaction by id")
        return rag_engine.get_transaction(transaction_id)
    except Exception as e:
        logger.error("Failed to fetch transaction: %s", e)
        raise


@mcp.tool()
def fraud_stats(
    since_hours: float | None = None,
    since_days: float | None = None,
    min_risk: float = 0.5,
) -> str:
    """Aggregate/quantitative statistics over a time window — use this for
    'how many', 'what rate', 'total', 'count' questions rather than listing
    transactions ('how many fraud today?', 'fraud rate in the last hour?',
    'total amount flagged this week?').
    Returns a JSON object: total_transactions, confirmed_fraud, suspected_high_risk
    (fraud_probability >= min_risk), fraud_rate and total_amount.
     - since_hours / since_days: the window ('today' -> since_days=1; 'last hour'
       -> since_hours=1). Omit both for all-time.
     - min_risk: threshold for 'suspected_high_risk' (default 0.5)."""
    try:
        logger.info("Computing fraud stats")
        return rag_engine.fraud_stats(since_hours=since_hours, since_days=since_days, min_risk=min_risk)
    except Exception as e:
        logger.error("Failed to compute fraud stats: %s", e)
        raise


if __name__ == "__main__":
    configure_logging()
    mcp.run(transport="sse")
