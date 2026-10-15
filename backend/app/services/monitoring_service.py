from collections import Counter
from collections import deque
from threading import Lock

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

from backend.app.services.prediction_store import get_all_predictions


class RuntimeMetrics:
    """Process-local counters; intentionally stores no transaction data."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: Counter[str] = Counter()
        self._latency_ms_total = 0.0
        self._latencies: deque[float] = deque(maxlen=10_000)

    def request(self, latency_ms: float, status_code: int = 200) -> None:
        with self._lock:
            self._counts["api_requests"] += 1
            self._latency_ms_total += latency_ms
            self._latencies.append(latency_ms)
            if status_code >= 500:
                self._counts["api_errors"] += 1

    def prediction(self, result: str) -> None:
        with self._lock:
            self._counts[result] += 1

    def snapshot(self) -> dict:
        with self._lock:
            total = self._counts["api_requests"]
            latencies = np.asarray(self._latencies, dtype=float)
            return {
                "api_requests": total,
                "average_latency_ms": round(self._latency_ms_total / total, 2)
                if total
                else 0.0,
                "prediction_successes": self._counts["prediction_successes"],
                "prediction_failures": self._counts["prediction_failures"],
                "fraud_predictions": self._counts["fraud_predictions"],
                "legitimate_predictions": self._counts["legitimate_predictions"],
                "api_errors": self._counts["api_errors"],
                "error_rate": round(self._counts["api_errors"] / total, 6) if total else 0.0,
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies.size else 0.0,
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies.size else 0.0,
            }


runtime_metrics = RuntimeMetrics()


def monitoring_summary() -> dict:
    rows = get_all_predictions()
    metrics = runtime_metrics.snapshot()
    total = len(rows)
    labeled = [row for row in rows if row.get("true_label") in (0, 1)]
    if labeled:
        actual = np.asarray([row["true_label"] for row in labeled], dtype=int)
        predicted = np.asarray([row["is_fraud"] for row in labeled], dtype=int)
        probabilities = np.asarray([row["fraud_probability"] for row in labeled], dtype=float)
        performance = {
            "labeled_predictions": len(labeled),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "pr_auc": float(average_precision_score(actual, probabilities)) if len(set(actual)) > 1 else None,
        }
    else:
        performance = {"labeled_predictions": 0, "precision": None, "recall": None, "pr_auc": None}
    return {
        **metrics,
        "predictions_stored": total,
        "fraud_predictions_stored": sum(bool(row["is_fraud"]) for row in rows),
        "fraud_prediction_rate": (
            sum(bool(row["is_fraud"]) for row in rows) / total if total else 0.0
        ),
        "performance": performance,
        "scope": "Process counters reset on restart; stored prediction history is read from JSONL.",
    }


def _psi(reference: np.ndarray, recent: np.ndarray) -> float:
    finite_reference = reference[np.isfinite(reference)]
    finite_recent = recent[np.isfinite(recent)]
    if not finite_reference.size or not finite_recent.size:
        return 0.0
    edges = np.unique(np.quantile(finite_reference, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected = np.histogram(finite_reference, bins=edges)[0] / finite_reference.size
    actual = np.histogram(finite_recent, bins=edges)[0] / finite_recent.size
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _ks_statistic(reference: np.ndarray, recent: np.ndarray) -> float:
    reference = np.sort(reference[np.isfinite(reference)])
    recent = np.sort(recent[np.isfinite(recent)])
    if not reference.size or not recent.size:
        return 0.0
    combined = np.sort(np.concatenate((reference, recent)))
    ref_cdf = np.searchsorted(reference, combined, side="right") / reference.size
    recent_cdf = np.searchsorted(recent, combined, side="right") / recent.size
    return float(np.max(np.abs(ref_cdf - recent_cdf)))


def drift_report() -> dict:
    import pandas as pd

    from backend.app.config.settings import PROJECT_ROOT

    reference_path = PROJECT_ROOT / "models" / "production" / "reference_sample.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Drift reference sample not found: {reference_path}")
    reference = pd.read_csv(reference_path)
    recent_records = get_all_predictions()
    recent_records = recent_records[-1000:]
    if not recent_records:
        return {"status": "insufficient_data", "sample_size": 0, "features": {}}
    recent = pd.DataFrame([row for row in recent_records])
    results = {}
    numeric_features = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for feature in numeric_features:
        if feature not in reference.columns or feature not in recent.columns:
            continue
        expected = pd.to_numeric(reference[feature], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(recent[feature], errors="coerce").to_numpy(dtype=float)
        psi = _psi(expected, current)
        results[feature] = {
            "psi": psi,
            "ks_statistic": _ks_statistic(expected, current),
            "status": "alert" if psi > 0.25 else "warning" if psi > 0.1 else "stable",
        }
    return {
        "status": "alert" if any(item["status"] == "alert" for item in results.values())
        else "warning" if any(item["status"] == "warning" for item in results.values())
        else "stable",
        "sample_size": len(recent),
        "reference_size": int(len(reference)),
        "features": results,
        "thresholds": {"warning_psi": 0.1, "alert_psi": 0.25},
    }
