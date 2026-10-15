from __future__ import annotations

import csv
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config import ROOT_DIR

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
RUNS_CSV = EXPERIMENTS_DIR / "runs.csv"
RUN_FIELDS = [
    "run_id",
    "timestamp",
    "model",
    "imbalance_method",
    "params",
    "metrics",
    "data_hash",
    "git_commit",
]


def log_run(
    *,
    model_name: str,
    imbalance_method: str,
    params: dict,
    metrics: dict,
    model,
    data_hash: str,
    git_commit: str | None = None,
    artifacts: dict[str, str | Path] | None = None,
) -> str:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "params.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    joblib.dump(model, run_dir / "model.joblib")
    for artifact_name, artifact_path in (artifacts or {}).items():
        if Path(artifact_name).name != artifact_name:
            raise ValueError(f"Artifact name must be a filename, not a path: {artifact_name!r}")
        source = Path(artifact_path)
        if not source.is_file():
            raise FileNotFoundError(f"Tracked artifact does not exist: {source}")
        shutil.copy2(source, run_dir / artifact_name)

    new_file = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "imbalance_method": imbalance_method,
                "params": json.dumps(params, sort_keys=True, default=str),
                "metrics": json.dumps(metrics, sort_keys=True, default=str),
                "data_hash": data_hash,
                "git_commit": git_commit or "",
            }
        )
    return run_id


def list_runs() -> list[dict]:
    if not RUNS_CSV.exists():
        return []
    with RUNS_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_runs() -> list[dict]:
    runs = list_runs()
    for row in runs:
        row["params"] = json.loads(row["params"])
        row["metrics"] = json.loads(row["metrics"])
    return sorted(runs, key=lambda row: row["metrics"].get("pr_auc", 0.0), reverse=True)


def load_run(run_id: str):
    model_path = EXPERIMENTS_DIR / run_id / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"No tracked model found for run '{run_id}'.")
    return joblib.load(model_path)
