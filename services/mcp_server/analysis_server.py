import logging

from mcp.server.fastmcp import FastMCP

from services.agent.sentence_transformer import SentenceTransformerModel
from services.tool.rag_tool import RAGQueryEngine
from shared.config_loader import config_loader

logger = logging.getLogger(__name__)
mcp = FastMCP("Analysis server", port=8004)

cfg = config_loader.load()
CORRELATION_MAP = cfg.correlation_analysis.features
THRESHOLDS = cfg.correlation_analysis.thresholds

@mcp.tool()
def interpret_fraud_features(v_features: dict) -> str:
    # Use this tool ONLY AFTER you have obtained specific numerical 'V' features (V1, V2, etc.) for a transaction.
    # This tool performs a mathematical analysis of the features against a correlation map to determine if they represent a fraud risk.
    # DO NOT use this tool to search for transactions or filter by amount;
    # it only interprets raw data already in your possession.
    significant_features = []

    for feature, value in v_features.items():
        if feature in CORRELATION_MAP:
            corr = CORRELATION_MAP[feature]

            is_anomaly = (corr < 0 and value < THRESHOLDS["critical_negative"]) or \
                         (corr > 0 and value > THRESHOLDS["critical_positive"])

            if is_anomaly:
                logger.info("Anomaly detected")
                impact = "High Risk" if abs(corr) > THRESHOLDS["high_impact_abs"] else "Moderate Risk"
                significant_features.append(
                    f"{feature} (Value: {value:.2f}, Corr: {corr:.2f}) -> {impact}"
                )

    return "Detected Anomalies: " + " | ".join(
        significant_features) if significant_features else "No anomalies detected."

if __name__ == "__main__":
    mcp.run(transport="sse")