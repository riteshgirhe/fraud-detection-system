import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from backend.app.ml import model_loader


def test_production_artifacts_load():
    model, threshold = model_loader.load_model_artifacts()
    assert callable(model.predict_proba)
    assert tuple(model.feature_names_in_) == model_loader.EXPECTED_FEATURES
    assert isinstance(threshold, float)
    assert 0 <= threshold <= 1


@pytest.mark.parametrize("threshold", ["0.2", True, -0.1, 1.1])
def test_invalid_threshold_is_rejected(monkeypatch, threshold):
    with TemporaryDirectory(prefix=".pytest-threshold-", dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        model_path = temp_path / "model.joblib"
        threshold_path = temp_path / "threshold.json"
        model_path.write_bytes(b"present")
        threshold_path.write_text(
            json.dumps({"decision_threshold": threshold}), encoding="utf-8"
        )
        monkeypatch.setattr(model_loader, "MODEL_PATH", model_path)
        monkeypatch.setattr(model_loader, "THRESHOLD_PATH", threshold_path)

        with pytest.raises(ValueError, match="Decision threshold"):
            model_loader.load_model_artifacts()
