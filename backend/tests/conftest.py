import pytest

from backend.app.ml.model_loader import load_model_artifacts
from backend.app.services import prediction_store
from backend.app.services.monitoring_service import runtime_metrics


@pytest.fixture(autouse=True)
def isolate_api_state(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_store, "PREDICTIONS_PATH", tmp_path / "predictions.jsonl")
    monkeypatch.setattr(prediction_store, "_records", None)
    load_model_artifacts.cache_clear()
    runtime_metrics._counts.clear()
    runtime_metrics._latency_ms_total = 0.0
    runtime_metrics._latencies.clear()
    yield
    load_model_artifacts.cache_clear()
