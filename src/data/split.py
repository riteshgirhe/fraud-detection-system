from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RANDOM_STATE
from src.features.engineer import engineer_features


def time_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, random_state: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    """Split by time using step ordering, leaving the latest segment for testing."""
    frame = engineer_features(df, keep_target=True).copy()
    frame = frame.sort_values("step").reset_index(drop=True)
    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    splits = {
        "train": frame.iloc[:train_end].copy(),
        "val": frame.iloc[train_end:val_end].copy(),
        "test": frame.iloc[val_end:].copy(),
    }
    return splits


def save_splits(df: pd.DataFrame, output_dir: str | Path = "data/processed") -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = time_split(df)
    for name, data in splits.items():
        path = output_dir / f"{name}.csv"
        data.to_csv(path, index=False)

    summary = {
        name: {
            "rows": int(frame.shape[0]),
            "fraud_rate": float(frame["isFraud"].mean()),
            "min_step": int(frame["step"].min()) if not frame.empty else None,
            "max_step": int(frame["step"].max()) if not frame.empty else None,
        }
        for name, frame in splits.items()
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return splits


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw = pd.read_csv(project_root / "data" / "interim" / "fraud_cleaned.csv")
    save_splits(raw, project_root / "data" / "processed")
