from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import INTERIM_DATA_PATH, RAW_DATA_PATH


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if df[col].dropna().empty:
            continue
        if df[col].dtype.kind == "i":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        else:
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_rows = len(df)
    summary = {
        "original_rows": original_rows,
        "dropped_rows": [],
        "final_rows": 0,
        "columns_removed": [],
        "dtypes": {},
    }

    before = len(df)
    df = df.drop_duplicates()
    dropped_duplicates = before - len(df)
    if dropped_duplicates:
        summary["dropped_rows"].append({"reason": "duplicate_rows", "count": int(dropped_duplicates)})

    before = len(df)
    df = df[df["amount"].notna()].copy()
    if len(df) != before:
        summary["dropped_rows"].append({"reason": "missing_amount", "count": int(before - len(df))})

    before = len(df)
    df = df[df["amount"] >= 0].copy()
    if len(df) != before:
        summary["dropped_rows"].append({"reason": "negative_amount", "count": int(before - len(df))})

    for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        before = len(df)
        df = df[df[col].notna()].copy()
        if len(df) != before:
            summary["dropped_rows"].append({"reason": f"missing_{col}", "count": int(before - len(df))})
        before = len(df)
        df = df[df[col] >= 0].copy()
        if len(df) != before:
            summary["dropped_rows"].append({"reason": f"negative_{col}", "count": int(before - len(df))})

    df["type"] = df["type"].astype(str).str.upper().str.strip()
    invalid_types = ~df["type"].isin({"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"})
    if invalid_types.any():
        removed = int(invalid_types.sum())
        summary["dropped_rows"].append({"reason": "invalid_transaction_type", "count": removed})
        df = df[~invalid_types].copy()

    if "isFlaggedFraud" in df.columns:
        summary["columns_removed"].append("isFlaggedFraud")
        df = df.drop(columns=["isFlaggedFraud"])

    df = _downcast_numeric(df)
    summary["final_rows"] = int(len(df))
    summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return df, summary


def run_cleaning(raw_path: str | Path = RAW_DATA_PATH, output_path: str | Path = INTERIM_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    cleaned_df, summary = clean_dataframe(df)
    cleaned_df.to_csv(output_path, index=False)
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cleaned_df, summary


if __name__ == "__main__":
    run_cleaning()
