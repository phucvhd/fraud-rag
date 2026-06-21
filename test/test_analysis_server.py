from services.mcp_server.analysis_server import interpret_fraud_features


def test_interpret_fraud_features_reports_ground_truth_fraud():
    result = interpret_fraud_features({"V17": -3.0}, is_fraud=True)
    assert "Ground Truth (DB label): FRAUD" in result


def test_interpret_fraud_features_reports_ground_truth_not_fraud():
    result = interpret_fraud_features({"V17": -3.0}, is_fraud=False)
    assert "Ground Truth (DB label): Not flagged as fraud" in result


def test_interpret_fraud_features_without_ground_truth_omits_line():
    result = interpret_fraud_features({"V17": -3.0})
    assert "Ground Truth" not in result


def test_interpret_fraud_features_no_anomaly_still_reports_ground_truth():
    result = interpret_fraud_features({"V17": 0.1}, is_fraud=True)
    assert "No anomalies detected by heuristic." in result
    assert "Ground Truth (DB label): FRAUD" in result
