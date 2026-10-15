from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import RANDOM_STATE
from src.preprocessing import load_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET = "isFraud"


def build_model(class_weight: str | None = "balanced") -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def build_pipeline(strategy: str):
    preprocessor = load_preprocessor(PROJECT_ROOT / "models" / "preprocessor.joblib")

    if strategy == "none":
        return (
            ImbPipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", build_model(class_weight="balanced")),
                ]
            ),
            {"sampler": "none", "class_weight": "balanced"},
        )

    if strategy == "undersample":
        return (
            ImbPipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
                    ("model", build_model(class_weight=None)),
                ]
            ),
            {"sampler": "RandomUnderSampler", "class_weight": None},
        )

    if strategy == "smote":
        return (
            ImbPipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("sampler", SMOTE(random_state=RANDOM_STATE)),
                    ("model", build_model(class_weight=None)),
                ]
            ),
            {"sampler": "SMOTE", "class_weight": None},
        )

    if strategy == "smote_tomek":
        return (
            ImbPipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("sampler", SMOTETomek(random_state=RANDOM_STATE)),
                    ("model", build_model(class_weight=None)),
                ]
            ),
            {"sampler": "SMOTETomek", "class_weight": None},
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def prepare_strategy_data(train_df: pd.DataFrame, strategy: str) -> tuple[pd.DataFrame, pd.Series]:
    if strategy in {"none", "undersample"}:
        return train_df.drop(columns=[TARGET], errors="ignore"), train_df[TARGET].astype(int)

    minority = train_df[train_df[TARGET] == 1]
    majority = train_df[train_df[TARGET] == 0].sample(n=min(len(train_df) // 3, 200000), random_state=RANDOM_STATE)
    sample_df = pd.concat([minority, majority], ignore_index=True)
    return sample_df.drop(columns=[TARGET], errors="ignore"), sample_df[TARGET].astype(int)


def evaluate_strategy(strategy: str, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    pipeline, metadata = build_pipeline(strategy)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_train)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "strategy": strategy,
        "sampler": metadata["sampler"],
        "class_weight": metadata["class_weight"],
        "train_rows": int(len(X_train)),
        "fraud_rows": int(y_train.sum()),
        "legit_rows": int((1 - y_train).sum()),
        "precision": float(precision_score(y_train, y_pred, zero_division=0)),
        "recall": float(recall_score(y_train, y_pred, zero_division=0)),
        "f1": float(f1_score(y_train, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_train, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def run_imbalance_comparison(
    train_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    train_path = Path(train_path) if train_path is not None else PROJECT_ROOT / "data" / "processed" / "train.csv"
    output_path = Path(output_path) if output_path is not None else PROJECT_ROOT / "reports" / "imbalance_comparison.json"

    train_df = pd.read_csv(train_path)
    strategies = ["none", "undersample", "smote", "smote_tomek"]
    comparison = {
        "note": "Resampling and class balancing were applied only to the training data. Validation and test sets were left untouched.",
        "train_data_rows": int(len(train_df)),
        "train_fraud_rate": float(train_df[TARGET].mean()),
        "strategies": [],
    }

    for strategy in strategies:
        strategy_X, strategy_y = prepare_strategy_data(train_df, strategy)
        comparison["strategies"].append(evaluate_strategy(strategy, strategy_X, strategy_y))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


if __name__ == "__main__":
    run_imbalance_comparison()
