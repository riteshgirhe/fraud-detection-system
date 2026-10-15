from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, REPORTS_DIR, VALIDATION_REPORT_PATH

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]
ALLOWED_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def _safe_json(value):
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    if isinstance(value, tuple):
        return [_safe_json(v) for v in value]
    return value


def validate_dataframe(df: pd.DataFrame) -> dict:
    report = {
        "file": str(RAW_DATA_PATH),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {col: int(value) for col, value in df.isna().sum().items() if value > 0},
        "negative_amounts": {
            col: int((df[col] < 0).sum()) for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        },
        "invalid_transaction_types": sorted(set(df["type"]) - ALLOWED_TYPES) if "type" in df.columns else [],
        "invalid_is_fraud_values": sorted(df.loc[~df["isFraud"].isin([0, 1]), "isFraud"].unique().tolist()) if "isFraud" in df.columns else [],
        "balance_consistency_violations": {},
        "critical_violations": [],
    }

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        report["critical_violations"].append(f"Missing required columns: {missing_columns}")

    if "type" in df.columns:
        invalid_types = sorted(set(df["type"]) - ALLOWED_TYPES)
        if invalid_types:
            report["critical_violations"].append(f"Invalid transaction type values: {invalid_types}")

    if "isFraud" in df.columns:
        invalid_labels = sorted(df.loc[~df["isFraud"].isin([0, 1]), "isFraud"].unique().tolist())
        if invalid_labels:
            report["critical_violations"].append(f"isFraud contains non-binary values: {invalid_labels}")

    for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if col in df.columns and (df[col] < 0).any():
            report["critical_violations"].append(f"Column '{col}' contains negative values")

    if {"oldbalanceOrg", "newbalanceOrig", "amount"}.issubset(df.columns):
        orig_balance_error = (df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]).abs()
        report["balance_consistency_violations"]["orig_balance_error_max"] = float(orig_balance_error.max())
        report["balance_consistency_violations"]["orig_balance_error_violations"] = int((orig_balance_error > 1e-5).sum())
        report["balance_consistency_violations"]["orig_balance_error_p99"] = float(orig_balance_error.quantile(0.99))

    if {"oldbalanceDest", "newbalanceDest", "amount"}.issubset(df.columns):
        dest_balance_error = (df["newbalanceDest"] - df["amount"] - df["oldbalanceDest"]).abs()
        report["balance_consistency_violations"]["dest_balance_error_max"] = float(dest_balance_error.max())
        report["balance_consistency_violations"]["dest_balance_error_violations"] = int((dest_balance_error > 1e-5).sum())
        report["balance_consistency_violations"]["dest_balance_error_p99"] = float(dest_balance_error.quantile(0.99))

    if report["critical_violations"]:
        report["status"] = "failed"
        raise ValueError("Critical validation issues detected: " + "; ".join(report["critical_violations"]))

    report["status"] = "passed"
    return report


def run_validation(raw_path: str | Path = RAW_DATA_PATH) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    report = validate_dataframe(df)
    VALIDATION_REPORT_PATH.write_text(json.dumps(_safe_json(report), indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_validation()
