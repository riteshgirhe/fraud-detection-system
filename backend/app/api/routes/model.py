from fastapi import APIRouter

from backend.app.ml.model_loader import load_model_metadata

router = APIRouter(tags=["model"])


@router.get("/model/info")
def model_info() -> dict:
    metadata = load_model_metadata()
    metrics = metadata.get("metrics", {})
    return {
        "version": metadata.get("version", "unknown"),
        "model": metadata.get("model", "unknown"),
        "threshold": metadata.get("threshold"),
        "metrics": metrics,
        "confusion_matrix": metrics.get("confusion_matrix"),
        "feature_importance": metrics.get("feature_importance", {}),
        "model_size_bytes": metadata.get("model_size_bytes"),
        "training_date": metadata.get("training_date"),
    }