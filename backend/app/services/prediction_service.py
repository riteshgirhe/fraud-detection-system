import math
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from backend.app.ml.model_loader import (
    EXPECTED_FEATURES,
    load_model_artifacts,
    load_model_metadata,
)
from backend.app.schemas.prediction import PredictionResponse, TransactionInput
from backend.app.services.prediction_store import append_prediction


def predict_transaction(transaction: TransactionInput) -> PredictionResponse:
    started = time.perf_counter()
    model, threshold = load_model_artifacts()
    metadata = load_model_metadata()
    row = transaction.model_dump()
    frame = pd.DataFrame(
        [{name: row[name] for name in EXPECTED_FEATURES}],
        columns=EXPECTED_FEATURES,
    )
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Loaded model has no positive fraud class (1).")
    probability = float(probabilities[0][classes.index(1)])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Model returned an invalid fraud probability.")
    is_fraud = probability >= threshold
    timestamp = datetime.now(timezone.utc)
    transaction_id = str(uuid.uuid4())
    latency_ms = (time.perf_counter() - started) * 1000
    result = PredictionResponse(
        transaction_id=transaction_id,
        type=transaction.type,
        amount=transaction.amount,
        fraud_probability=probability,
        decision_threshold=threshold,
        is_fraud=is_fraud,
        prediction="Fraud Detected" if is_fraud else "Legitimate Transaction",
        model_version=str(metadata.get("version", "unknown")),
        timestamp=timestamp,
        latency_ms=latency_ms,
    )
    append_prediction(
        {
            **row,
            "transaction_id": transaction_id,
            "fraud_probability": probability,
            "decision_threshold": threshold,
            "is_fraud": is_fraud,
            "prediction": result.prediction,
            "model_version": result.model_version,
            "timestamp": timestamp.isoformat(),
            "latency_ms": latency_ms,
            "true_label": None,
        }
    )
    return result
