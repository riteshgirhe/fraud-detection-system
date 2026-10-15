from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import prediction_service

client = TestClient(app)
PAYLOAD = {
    "step": 1,
    "type": "PAYMENT",
    "amount": 42.5,
    "oldbalanceOrg": 100.0,
    "newbalanceOrig": 57.5,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 42.5,
}


class FixedModel:
    classes_ = [0, 1]

    def predict_proba(self, frame):
        assert list(frame.columns) == [
            "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest",
        ]
        return [[0.15, 0.85]]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cors_allows_configured_frontend_only():
    allowed = client.options(
        "/api/v1/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    blocked = client.options(
        "/api/v1/predict",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in blocked.headers


def test_valid_prediction_request(monkeypatch):
    monkeypatch.setattr(prediction_service, "load_model_artifacts", lambda: (FixedModel(), 0.2))
    response = client.post("/api/v1/predict", json=PAYLOAD)
    assert response.status_code == 200
    result = response.json()
    assert result["type"] == PAYLOAD["type"]
    assert result["amount"] == PAYLOAD["amount"]
    assert result["fraud_probability"] == 0.85
    assert result["decision_threshold"] == 0.2
    assert result["is_fraud"] is True
    assert result["prediction"] == "Fraud Detected"
    assert result["transaction_id"]
    assert result["model_version"]
    assert result["timestamp"]
    assert result["latency_ms"] >= 0


def test_below_threshold_is_legitimate(monkeypatch):
    class LegitimateModel(FixedModel):
        def predict_proba(self, frame):
            return [[0.95, 0.05]]

    monkeypatch.setattr(prediction_service, "load_model_artifacts", lambda: (LegitimateModel(), 0.2))
    response = client.post("/api/v1/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json()["is_fraud"] is False
    assert response.json()["prediction"] == "Legitimate Transaction"


def test_invalid_type_and_negative_value_are_rejected():
    invalid_type = {**PAYLOAD, "type": "WIRE"}
    negative_amount = {**PAYLOAD, "amount": -1}
    boolean_amount = {**PAYLOAD, "amount": True}
    assert client.post("/api/v1/predict", json=invalid_type).status_code == 422
    assert client.post("/api/v1/predict", json=negative_amount).status_code == 422
    assert client.post("/api/v1/predict", json=boolean_amount).status_code == 422


def test_model_failure_is_reported_as_service_unavailable(monkeypatch):
    def broken_loader():
        raise FileNotFoundError("artifact is unavailable")

    monkeypatch.setattr(prediction_service, "load_model_artifacts", broken_loader)
    response = client.post("/api/v1/predict", json=PAYLOAD)
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction service unavailable"}


def test_monitoring_metrics_endpoint():
    response = client.get("/api/v1/monitoring/metrics")
    assert response.status_code == 200
    assert "api_requests" in response.json()
    assert "average_latency_ms" in response.json()


def test_batch_history_and_label_endpoints(monkeypatch):
    monkeypatch.setattr(prediction_service, "load_model_artifacts", lambda: (FixedModel(), 0.2))
    batch = client.post(
        "/api/v1/predict/batch",
        json={"transactions": [PAYLOAD, {**PAYLOAD, "amount": 80.0}]},
    )
    assert batch.status_code == 200
    results = batch.json()
    assert len(results) == 2
    assert [result["amount"] for result in results] == [42.5, 80.0]

    history = client.get("/api/v1/predictions?offset=0&limit=1")
    assert history.status_code == 200
    assert history.json()["total"] == 2
    assert len(history.json()["items"]) == 1

    transaction_id = results[0]["transaction_id"]
    labeled = client.patch(
        f"/api/v1/predictions/{transaction_id}/label",
        json={"true_label": 1},
    )
    assert labeled.status_code == 200
    assert labeled.json()["true_label"] == 1

    labeled_history = client.get("/api/v1/predictions?labeled=true")
    assert labeled_history.status_code == 200
    assert labeled_history.json()["total"] == 1
    missing = client.patch(
        "/api/v1/predictions/missing/label",
        json={"true_label": 0},
    )
    assert missing.status_code == 404


def test_model_reports_and_monitoring_routes():
    assert client.get("/api/v1/model/info").status_code == 200
    assert client.get("/api/v1/reports/eda").status_code == 200
    assert client.get("/api/v1/reports/comparison").status_code == 200
    summary = client.get("/api/v1/monitoring/summary")
    drift = client.get("/api/v1/monitoring/drift")
    assert summary.status_code == 200
    assert summary.json()["predictions_stored"] == 0
    assert drift.status_code == 200
    assert drift.json()["sample_size"] == 0
