
from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _artifact_path(env_var: str, filename: str) -> Path:
    """Resolve artifact paths from the project root, regardless of CWD."""
    configured_path = Path(os.getenv(env_var, f"models/production/{filename}"))
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    return configured_path.resolve()


MODEL_PATH = _artifact_path("FRAUD_MODEL_PATH", "fraud_detection_pipeline.joblib")
THRESHOLD_PATH = _artifact_path("FRAUD_THRESHOLD_PATH", "decision_threshold.json")

FRONTEND_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").split(",")
    if origin.strip()
)
API_PREFIX = "/api/v1"

APP_NAME = "Fraud Detection API"
APP_VERSION = "1.0.0"
