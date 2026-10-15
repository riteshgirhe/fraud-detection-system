from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import COST_FN, COST_FP, RANDOM_STATE, ROOT_DIR
from src.features.engineer import FeatureEngineerTransformer, get_feature_columns
from src.tracking.tracker import log_run

TARGET = "isFraud"
MODEL_FEATURES = [name for name in get_feature_columns() if name != "is_merchant_dest"]
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
VALIDATION_PATH = ROOT_DIR / "data" / "processed" / "val.csv"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
REPORTS_DIR = ROOT_DIR / "reports"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
MAX_TRAIN_NEGATIVES = 30_000
API_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def _dataset_hash() -> str:
    digest = hashlib.sha256()
    with (ROOT_DIR / "data" / "raw" / "raw_data.csv").open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def _read_training_sample(path: Path = TRAIN_PATH, max_negatives: int = MAX_TRAIN_NEGATIVES) -> pd.DataFrame:
    usecols = [*MODEL_FEATURES, TARGET]
    total_rows = _count_rows(path)
    negative_rate = max_negatives / max(total_rows, 1)
    negatives: list[pd.DataFrame] = []
    positives: list[pd.DataFrame] = []

    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=200_000)):
        positives.append(chunk.loc[chunk[TARGET] == 1])
        majority = chunk.loc[chunk[TARGET] == 0]
        sample_count = min(len(majority), max(1, round(len(majority) * negative_rate)))
        negatives.append(majority.sample(n=sample_count, random_state=RANDOM_STATE + chunk_index))

    sampled = pd.concat([*positives, *negatives], ignore_index=True)
    return sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def _read_evaluation_sample(path: Path, negative_limit: int | None) -> pd.DataFrame:
    if negative_limit is None:
        return pd.read_csv(path, usecols=[*MODEL_FEATURES, TARGET])

    usecols = [*MODEL_FEATURES, TARGET]
    positives: list[pd.DataFrame] = []
    negatives: list[pd.DataFrame] = []
    total_rows = _count_rows(path)
    negative_rate = negative_limit / max(total_rows, 1)

    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=200_000)):
        positives.append(chunk.loc[chunk[TARGET] == 1])
        majority = chunk.loc[chunk[TARGET] == 0]
        take = min(len(majority), round(len(majority) * negative_rate))
        if take:
            negatives.append(majority.sample(n=take, random_state=RANDOM_STATE + chunk_index))
    selected = pd.concat([*positives, *negatives], ignore_index=True)
    return selected.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def _build_classifier(family: str, positive_weight: float):
    if family == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if family == "xgboost":
        return XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            scale_pos_weight=positive_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=4,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported model family: {family}")


def _make_pipeline(family: str, strategy: str, y_train: pd.Series):
    positive = max(int(y_train.sum()), 1)
    negative = max(int((y_train == 0).sum()), 1)
    positive_weight = negative / positive
    sampler = {
        "none": "passthrough",
        "undersample": RandomUnderSampler(random_state=RANDOM_STATE),
        "smote": SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, positive - 1)),
        "smote_tomek": SMOTETomek(
            random_state=RANDOM_STATE,
            smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, positive - 1)),
        ),
    }.get(strategy)
    if sampler is None:
        raise ValueError(f"Unsupported imbalance strategy: {strategy}")

    classifier = _build_classifier(family, positive_weight)
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("sampler", sampler),
            ("classifier", classifier),
        ]
    )


def _metrics(y_true, probabilities, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _score_candidate(model, X, y) -> tuple[dict, float]:
    started = time.perf_counter()
    probabilities = model.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - started
    result = _metrics(y, probabilities)
    result["inference_ms_per_10000"] = float(elapsed * 1000 * 10_000 / max(len(X), 1))
    serialized = io.BytesIO()
    joblib.dump(model, serialized)
    result["model_size_bytes"] = int(serialized.tell())
    return result, probabilities


def run_baseline(train_sample: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.Series, data_hash: str) -> dict:
    y_train = train_sample[TARGET].astype(int)
    X_train = train_sample.drop(columns=[TARGET])
    all_legitimate = _metrics(y_val, np.zeros(len(y_val), dtype=np.float32))

    model = _make_pipeline("logistic_regression", "none", y_train)
    model.fit(X_train, y_train)
    metrics, _ = _score_candidate(model, X_val, y_val)
    log_run(
        model_name="logistic_regression_baseline",
        imbalance_method="class_weight_balanced",
        params={"class_weight": "balanced", "max_iter": 500},
        metrics=metrics,
        model=model,
        data_hash=data_hash,
    )
    naive_model = DummyClassifier(strategy="constant", constant=0).fit(X_train, y_train)
    log_run(
        model_name="all_legitimate_baseline",
        imbalance_method="none",
        params={"strategy": "constant", "constant": 0},
        metrics=all_legitimate,
        model=naive_model,
        data_hash=data_hash,
    )
    report = {
        "evaluation_data": "time-based validation split",
        "training_rows": int(len(train_sample)),
        "training_fraud_rate": float(y_train.mean()),
        "all_legitimate_baseline": all_legitimate,
        "class_weighted_logistic_regression": metrics,
    }
    (REPORTS_DIR / "baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_model_comparison(train_sample, X_val, y_val, data_hash: str) -> list[dict]:
    X_train = train_sample.drop(columns=[TARGET])
    y_train = train_sample[TARGET].astype(int)
    candidate_specs = [
        ("logistic_regression", "none"),
        ("logistic_regression", "undersample"),
        ("logistic_regression", "smote"),
        ("logistic_regression", "smote_tomek"),
        ("random_forest", "none"),
        ("xgboost", "none"),
    ]
    results: list[dict] = []
    for family, strategy in candidate_specs:
        model = _make_pipeline(family, strategy, y_train)
        model.fit(X_train, y_train)
        metrics, _ = _score_candidate(model, X_val, y_val)
        metrics.update(
            {
                "model": family,
                "imbalance_method": strategy,
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_val)),
            }
        )
        run_id = log_run(
            model_name=family,
            imbalance_method=strategy,
            params={"pipeline": model.get_params(deep=False), "training_sample_rows": len(X_train)},
            metrics=metrics,
            model=model,
            data_hash=data_hash,
        )
        metrics["run_id"] = run_id
        results.append(metrics)

    cv_scores = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for family in ("logistic_regression", "random_forest", "xgboost"):
        model = _make_pipeline(family, "none", y_train)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="average_precision",
            n_jobs=1,
        )
        cv_scores[family] = {
            "fold_pr_auc": [float(score) for score in scores],
            "mean_pr_auc": float(np.mean(scores)),
            "std_pr_auc": float(np.std(scores)),
            "folds": 3,
        }

    report = {
        "validation_rows": int(len(X_val)),
        "training_sample_rows": int(len(X_train)),
        "validation_metrics": results,
        "stratified_cv": cv_scores,
        "selection_note": "Validation metrics use the full time-ordered validation split; CV is stratified and restricted to the training sample.",
    }
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return results


def _parameter_distributions(family: str, positive_weight: float) -> dict:
    if family == "xgboost":
        return {
            "classifier__n_estimators": [100, 150, 250, 350],
            "classifier__max_depth": [3, 4, 5, 7],
            "classifier__learning_rate": [0.02, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__gamma": [0, 0.1, 0.5],
            "classifier__scale_pos_weight": [positive_weight * 0.5, positive_weight, positive_weight * 2],
        }
    if family == "random_forest":
        return {
            "classifier__n_estimators": [100, 150, 250],
            "classifier__max_depth": [8, 12, 18, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.7],
        }
    return {
        "classifier__C": [0.1, 0.3, 1.0, 3.0, 10.0],
    }


def tune_top_models(train_sample, validation_sample, comparison_results, data_hash: str) -> list[dict]:
    ranked = sorted(comparison_results, key=lambda row: row["pr_auc"], reverse=True)
    families: list[str] = []
    for row in ranked:
        if row["model"] not in families:
            families.append(row["model"])
        if len(families) == 2:
            break

    X_train = train_sample.drop(columns=[TARGET])
    y_train = train_sample[TARGET].astype(int)
    X_val = validation_sample.drop(columns=[TARGET])
    y_val = validation_sample[TARGET].astype(int)
    positive_weight = max(int((y_train == 0).sum()), 1) / max(int(y_train.sum()), 1)
    results = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for family in families:
        pipeline = _make_pipeline(family, "none", y_train)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=_parameter_distributions(family, positive_weight),
            n_iter=5,
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        metrics, _ = _score_candidate(search.best_estimator_, X_val, y_val)
        entry = {
            "model": family,
            "best_params": search.best_params_,
            "cv_best_pr_auc": float(search.best_score_),
            "validation": metrics,
        }
        log_run(
            model_name=family,
            imbalance_method="randomized_search_class_weighted",
            params=search.best_params_,
            metrics={"cv_best_pr_auc": float(search.best_score_), **metrics},
            model=search.best_estimator_,
            data_hash=data_hash,
        )
        results.append({"entry": entry, "model": search.best_estimator_})

    report = [
        {"model": result["entry"]["model"], "best_params": result["entry"]["best_params"],
         "cv_best_pr_auc": result["entry"]["cv_best_pr_auc"], "validation": result["entry"]["validation"]}
        for result in results
    ]
    (REPORTS_DIR / "tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return results


def choose_threshold(y_true, probabilities) -> tuple[float, list[dict]]:
    rows = []
    best_threshold = 0.0
    best_f2 = -1.0
    for threshold in np.linspace(0.0, 1.0, 201):
        predictions = probabilities >= threshold
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        f2 = fbeta_score(y_true, predictions, beta=2, zero_division=0)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "f2": float(f2),
            }
        )
        if f2 > best_f2:
            best_f2 = float(f2)
            best_threshold = float(threshold)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "threshold_curve.csv", index=False)
    return best_threshold, rows


def _next_version(registry: Path) -> str:
    versions = []
    for child in registry.glob("v*"):
        if child.is_dir() and child.name[1:].isdigit():
            versions.append(int(child.name[1:]))
    return f"v{max(versions, default=0) + 1}"


def promote(version: str, production_dir: Path | None = None) -> None:
    if len(version) < 2 or version[0] != "v" or not version[1:].isdigit():
        raise ValueError(f"Invalid registry version: {version!r}")
    registry_dir = ROOT_DIR / "models" / "registry" / version
    production_dir = production_dir or ROOT_DIR / "models" / "production"
    required = ["model.joblib", "fraud_detection_pipeline.joblib", "decision_threshold.json", "metadata.json", "reference_sample.csv"]
    missing = [name for name in required if not (registry_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot promote {version}; missing artifacts: {missing}")
    production_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(registry_dir / name, production_dir / name)


def rollback(version: str) -> None:
    promote(version)


def _test_report(y_test, probabilities, threshold: float, family: str, metadata: dict) -> dict:
    metrics = _metrics(y_test, probabilities, threshold)
    cm = metrics["confusion_matrix"]
    false_negative_rate = cm["fn"] / max(cm["fn"] + cm["tp"], 1)
    cost_fp = float(metadata.get("cost_fp", COST_FP))
    cost_fn = float(metadata.get("cost_fn", COST_FN))
    metrics.update(
        {
            "model": family,
            "false_negative_rate": float(false_negative_rate),
            "expected_cost": float(cm["fp"] * cost_fp + cm["fn"] * cost_fn),
            "cost_assumptions": {"cost_fp": cost_fp, "cost_fn": cost_fn},
            "limitations": "PaySim is synthetic; time-based splitting and repeated model selection may still produce optimistic estimates.",
        }
    )
    return metrics


def run_workflow() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "models" / "registry").mkdir(parents=True, exist_ok=True)
    data_hash = _dataset_hash()
    training_sample = _read_training_sample()
    y_train = training_sample[TARGET].astype(int)

    validation_sample = _read_evaluation_sample(VALIDATION_PATH, None)
    X_val = validation_sample.drop(columns=[TARGET])
    y_val = validation_sample[TARGET].astype(int)

    baseline = run_baseline(training_sample, X_val, y_val, data_hash)
    comparisons = run_model_comparison(training_sample, X_val, y_val, data_hash)
    tuned = tune_top_models(training_sample, validation_sample, comparisons, data_hash)
    max_latency = max(item["entry"]["validation"]["inference_ms_per_10000"] for item in tuned)
    max_size = max(item["entry"]["validation"]["model_size_bytes"] for item in tuned)
    for item in tuned:
        metrics = item["entry"]["validation"]
        selection_components = {
            "pr_auc": 0.65 * metrics["pr_auc"],
            "f2": 0.20 * metrics["f2"],
            "precision": 0.05 * metrics["precision"],
            "latency": 0.05 * (1 - metrics["inference_ms_per_10000"] / max(max_latency, 1e-9)),
            "model_size": 0.05 * (1 - metrics["model_size_bytes"] / max(max_size, 1)),
        }
        item["selection_score"] = sum(selection_components.values())
        item["selection_components"] = selection_components
        item["entry"]["validation"]["selection_score"] = item["selection_score"]
        item["entry"]["selection_components"] = selection_components
    tuned.sort(key=lambda item: item["selection_score"], reverse=True)
    (REPORTS_DIR / "tuning.json").write_text(
        json.dumps(
            [
                {
                    "model": item["entry"]["model"],
                    "best_params": item["entry"]["best_params"],
                    "cv_best_pr_auc": item["entry"]["cv_best_pr_auc"],
                    "validation": item["entry"]["validation"],
                    "selection_components": item["selection_components"],
                }
                for item in tuned
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    selected = tuned[0]
    selected_model = selected["model"]
    family = selected["entry"]["model"]

    validation_probabilities = selected_model.predict_proba(X_val)[:, 1]
    threshold, _ = choose_threshold(y_val, validation_probabilities)
    threshold_metrics = _metrics(y_val, validation_probabilities, threshold)
    log_run(
        model_name=family,
        imbalance_method="selected_validation_threshold",
        params={"threshold_rule": "maximize F2", "threshold": threshold},
        metrics=threshold_metrics,
        model=selected_model,
        data_hash=data_hash,
        artifacts={"threshold_curve.csv": REPORTS_DIR / "threshold_curve.csv"},
    )

    # The test set is first loaded here, after model and threshold selection.
    test_sample = _read_evaluation_sample(TEST_PATH, None)
    X_test = test_sample.drop(columns=[TARGET])
    y_test = test_sample[TARGET].astype(int)
    test_probabilities = selected_model.predict_proba(X_test)[:, 1]

    classifier = selected_model.named_steps["classifier"]
    scaler = selected_model.named_steps["scaler"]
    feature_transformer = FeatureEngineerTransformer(include_merchant_feature=False)
    raw_training_input = training_sample[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].copy()
    types = training_sample[[f"type_{kind}" for kind in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]]
    raw_training_input.insert(1, "type", types.idxmax(axis=1).str.removeprefix("type_"))
    feature_transformer.fit(raw_training_input, y_train)
    production_pipeline = ImbPipeline(
        steps=[
            ("feature_engineering", feature_transformer),
            ("scaler", scaler),
            ("classifier", classifier),
        ]
    )

    # The complete evaluation report records the final chosen threshold and held-out test metrics.
    evaluation_metadata = {"cost_fp": 1.0, "cost_fn": 10.0}
    final_metrics = _test_report(y_test, test_probabilities, threshold, family, evaluation_metadata)
    final_metrics.update(
        {
            "validation": threshold_metrics,
            "validation_threshold": threshold,
            "training_sample_rows": int(len(training_sample)),
            "test_rows": int(len(test_sample)),
            "selected_params": selected["entry"]["best_params"],
            "model_selection": {
                "score": selected["selection_score"],
                "components": selected["selection_components"],
                "candidate_count": len(tuned),
            },
        }
    )
    false_negative_mask = (y_test.to_numpy() == 1) & (test_probabilities < threshold)
    false_negative_rows = test_sample.loc[false_negative_mask]
    false_negative_types = {}
    if not false_negative_rows.empty:
        type_columns = [f"type_{kind}" for kind in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]
        false_negative_types = (
            false_negative_rows[type_columns]
            .idxmax(axis=1)
            .str.removeprefix("type_")
            .value_counts()
            .to_dict()
        )
    final_metrics["false_negative_analysis"] = {
        "count": int(false_negative_mask.sum()),
        "mean_amount": float(false_negative_rows["amount"].mean()) if not false_negative_rows.empty else None,
        "median_amount": float(false_negative_rows["amount"].median()) if not false_negative_rows.empty else None,
        "transaction_types": false_negative_types,
    }
    feature_importance = getattr(classifier, "feature_importances_", None)
    if feature_importance is not None:
        final_metrics["feature_importance"] = {
            feature: float(value)
            for feature, value in zip(MODEL_FEATURES, feature_importance, strict=True)
        }
    (REPORTS_DIR / "final_evaluation.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    comparison_rows = []
    for result in comparisons:
        comparison_rows.append(result)
    comparison_md = [
        "# Model comparison",
        "",
        "| Model | Imbalance method | Validation PR-AUC | Precision | Recall | F2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(comparison_rows, key=lambda item: item["pr_auc"], reverse=True):
        comparison_md.append(
            f"| {row['model']} | {row['imbalance_method']} | {row['pr_auc']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f2']:.6f} |"
        )
    comparison_md += [
        "",
        "## Tuned candidates",
        "",
        "| Model | CV PR-AUC | Validation PR-AUC | F2 | Latency ms / 10k | Size bytes | Selection score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in tuned:
        entry = item["entry"]
        metrics = entry["validation"]
        comparison_md.append(
            f"| {entry['model']} | {entry['cv_best_pr_auc']:.6f} | {metrics['pr_auc']:.6f} | "
            f"{metrics['f2']:.6f} | {metrics['inference_ms_per_10000']:.3f} | "
            f"{metrics['model_size_bytes']} | {item['selection_score']:.6f} |"
        )
    comparison_md += [
        "",
        "## Selection rationale",
        "",
        f"Selected {family} after randomized CV tuning. Final ranking combines validation PR-AUC, "
        "F2, precision, inference latency, and serialized model size. The threshold was then "
        "chosen to maximize F2 on the validation split; the held-out test set remained unused "
        "until selection was complete.",
    ]
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(comparison_md), encoding="utf-8")

    version = _next_version(ROOT_DIR / "models" / "registry")
    version_dir = ROOT_DIR / "models" / "registry" / version
    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(production_pipeline, version_dir / "model.joblib")
    shutil.copy2(version_dir / "model.joblib", version_dir / "fraud_detection_pipeline.joblib")
    (version_dir / "decision_threshold.json").write_text(
        json.dumps({"decision_threshold": threshold}, indent=2), encoding="utf-8"
    )
    reference = training_sample.drop(columns=[TARGET]).head(5000).copy()
    reference.to_csv(version_dir / "reference_sample.csv", index=False)

    model_size_bytes = (version_dir / "model.joblib").stat().st_size
    metadata = {
        "version": version,
        "model": family,
        "threshold": threshold,
        "threshold_rule": "maximize F2 on validation split",
        "feature_list": API_COLUMNS,
        "engineered_features": MODEL_FEATURES,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "metrics": final_metrics,
        "model_selection": final_metrics["model_selection"],
        "dataset_hash": data_hash,
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "model_size_bytes": model_size_bytes,
        "training_sample_rows": int(len(training_sample)),
        "reference_sample_rows": int(len(reference)),
        "excluded_features": {
            "is_merchant_dest": "Not used by this trained model because the API input schema does not include nameDest."
        },
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Compare saved raw-schema predictions with predictions from the equivalent engineered features.
    roundtrip_rows = test_sample.head(100)
    if len(roundtrip_rows):
        raw_roundtrip = roundtrip_rows[["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].copy()
        raw_types = roundtrip_rows[[f"type_{kind}" for kind in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]]
        raw_roundtrip.insert(1, "type", raw_types.idxmax(axis=1).str.removeprefix("type_"))
        expected = classifier.predict_proba(scaler.transform(roundtrip_rows[MODEL_FEATURES]))[:, 1]
        reloaded = joblib.load(version_dir / "model.joblib")
        actual = reloaded.predict_proba(raw_roundtrip)[:, 1]
        if not np.array_equal(expected, actual):
            raise AssertionError("Reloaded production artifact did not reproduce evaluation probabilities exactly.")
        metadata["round_trip_verified_rows"] = int(len(roundtrip_rows))
        (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        raise RuntimeError("No eligible rows found for production-artifact round-trip verification.")

    promote(version)
    return {
        "version": version,
        "selected_model": family,
        "threshold": threshold,
        "validation_f2": threshold_metrics["f2"],
        "test_metrics": final_metrics,
        "baseline": baseline,
        "comparison_count": len(comparisons),
    }


if __name__ == "__main__":
    print(json.dumps(run_workflow(), indent=2))
