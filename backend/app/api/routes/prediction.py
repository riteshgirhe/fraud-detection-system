import logging

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.prediction import (
    BatchPredictionInput,
    LabelInput,
    PredictionResponse,
    TransactionInput,
)
from backend.app.services.monitoring_service import runtime_metrics
from backend.app.services.prediction_store import attach_label, list_predictions
from backend.app.services.prediction_service import predict_transaction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionInput) -> PredictionResponse:
    try:
        result = predict_transaction(payload)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        runtime_metrics.prediction("prediction_failures")
        logger.error("prediction_unavailable error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Prediction service unavailable") from exc
    except Exception as exc:
        runtime_metrics.prediction("prediction_failures")
        logger.error("prediction_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    runtime_metrics.prediction("prediction_successes")
    runtime_metrics.prediction(
        "fraud_predictions" if result.is_fraud else "legitimate_predictions"
    )
    return result


@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(payload: BatchPredictionInput) -> list[PredictionResponse]:
    results = []
    for transaction in payload.transactions:
        results.append(predict(transaction))
    return results


@router.get("/predictions")
def predictions(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    is_fraud: bool | None = None,
    labeled: bool | None = None,
) -> dict:
    return list_predictions(offset=offset, limit=limit, is_fraud=is_fraud, labeled=labeled)


@router.patch("/predictions/{transaction_id}/label")
def label_prediction(transaction_id: str, payload: LabelInput) -> dict:
    try:
        return attach_label(transaction_id, payload.true_label)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prediction not found") from exc
