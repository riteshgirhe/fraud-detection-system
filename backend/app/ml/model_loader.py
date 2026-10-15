
import json
from functools import lru_cache

import joblib

from backend.app.config.settings import MODEL_PATH, THRESHOLD_PATH

EXPECTED_FEATURES = (
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)


@lru_cache(maxsize=1)
def load_model_artifacts():
    """
    Load the trained fraud detection pipeline and decision threshold once.
    Returns:
        tuple: (model_pipeline, decision_threshold)
    """

    # Check that the model file exists
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Check that the threshold file exists
    if not THRESHOLD_PATH.is_file():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    # Load the decision threshold
    with THRESHOLD_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if "decision_threshold" not in config:
        raise ValueError(
            "Threshold JSON must contain 'decision_threshold'."
        )

    threshold = config["decision_threshold"]

    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float)
    ):
        raise ValueError("Decision threshold must be numeric.")

    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Decision threshold must be between 0 and 1."
        )

    # Load the complete preprocessing + model pipeline after validating configuration.
    model_pipeline = joblib.load(MODEL_PATH)

    expected_features = getattr(model_pipeline, "feature_names_in_", None)
    if expected_features is None or tuple(expected_features) != EXPECTED_FEATURES:
        raise ValueError(
            "Loaded model input columns do not match the transaction API schema."
        )

    # Verify that the loaded model supports probability predictions
    if not callable(getattr(model_pipeline, "predict_proba", None)):
        raise TypeError(
            "Loaded model does not support predict_proba()."
        )

    return model_pipeline, threshold


def load_model_metadata() -> dict:
    metadata_path = MODEL_PATH.parent / "metadata.json"
    if not metadata_path.is_file():
        return {"version": "unknown", "metrics": {}}
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")
    return metadata
