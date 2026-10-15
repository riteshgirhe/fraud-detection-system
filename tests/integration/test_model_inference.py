from backend.app.ml.model_loader import load_model_artifacts
from backend.app.schemas.prediction import TransactionInput
from backend.app.services import prediction_store
from backend.app.services.prediction_service import predict_transaction


def test_saved_model_predicts_a_transaction(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_store, "PREDICTIONS_PATH", tmp_path / "predictions.jsonl")
    monkeypatch.setattr(prediction_store, "_records", None)
    load_model_artifacts.cache_clear()
    transaction = TransactionInput(
        step=1,
        type="PAYMENT",
        amount=42.5,
        oldbalanceOrg=100.0,
        newbalanceOrig=57.5,
        oldbalanceDest=0.0,
        newbalanceDest=42.5,
    )
    result = predict_transaction(transaction)
    assert result.type == transaction.type
    assert result.amount == transaction.amount
    assert 0.0 <= result.fraud_probability <= 1.0
    assert 0.0 <= result.decision_threshold <= 1.0
    assert result.is_fraud == (result.fraud_probability >= result.decision_threshold)
