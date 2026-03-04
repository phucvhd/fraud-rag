import uvicorn
from mcp.server.fastmcp import FastMCP

from shared.config_loader import config_loader

mcp = FastMCP("Fraud_Interpreter", port=8003)

cfg = config_loader.load()
CORRELATION_MAP = cfg.correlation_analysis.features
THRESHOLDS = cfg.correlation_analysis.thresholds


@mcp.tool()
def interpret_fraud_features(v_features: dict) -> str:
    significant_features = []

    for feature, value in v_features.items():
        if feature in CORRELATION_MAP:
            corr = CORRELATION_MAP[feature]

            is_anomaly = (corr < 0 and value < THRESHOLDS["critical_negative"]) or \
                         (corr > 0 and value > THRESHOLDS["critical_positive"])

            if is_anomaly:
                impact = "High Risk" if abs(corr) > THRESHOLDS["high_impact_abs"] else "Moderate Risk"
                significant_features.append(
                    f"{feature} (Value: {value:.2f}, Corr: {corr:.2f}) -> {impact}"
                )

    return "Detected Anomalies: " + " | ".join(
        significant_features) if significant_features else "No anomalies detected."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")