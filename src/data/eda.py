from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EDA_BY_HOUR_PATH, EDA_BY_TYPE_PATH, EDA_MD_PATH, EDA_REPORT_PATH, RAW_DATA_PATH


def _as_serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_as_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _as_serializable(v) for k, v in value.items()}
    return value


def _compute_balance_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["newbalanceDest"] - df["amount"] - df["oldbalanceDest"]
    return df


def _safe_table(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def build_eda_report(df: pd.DataFrame) -> dict:
    report_df = _compute_balance_errors(df)

    fraud_ratio = float(report_df["isFraud"].mean())
    fraud_by_type = (
        report_df.groupby("type", as_index=False)
        .agg(rows=("isFraud", "size"), fraud_count=("isFraud", "sum"), fraud_rate=("isFraud", "mean"))
        .sort_values(["fraud_count", "rows"], ascending=[False, False])
        .round({"fraud_rate": 6})
    )
    fraud_by_hour = (
        report_df.assign(hour_of_day=report_df["step"] % 24)
        .groupby("hour_of_day", as_index=False)
        .agg(fraud_count=("isFraud", "sum"), total_rows=("isFraud", "size"), fraud_rate=("isFraud", "mean"))
        .sort_values("hour_of_day")
    )

    report_df["log_amount"] = np.log1p(report_df["amount"])
    bins = np.linspace(report_df["log_amount"].min(), report_df["log_amount"].max(), 25)
    report_df["amount_bin"] = pd.cut(report_df["log_amount"], bins, include_lowest=True)
    amount_distribution = (
        report_df.groupby(["amount_bin", "isFraud"], observed=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"isFraud": "label"})
    )

    balance_corr = report_df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "isFraud"]].corr().round(4)
    corr_records = _safe_table(balance_corr.reset_index().rename(columns={"index": "variable"}))

    leakage_risks = [
        "Raw account identifiers (nameOrig, nameDest) should not be used as model features because they encode account identity rather than behavioral risk.",
        "The balance post-transaction columns can reflect the same transactional outcome as the target and may create leakage if used without careful feature design.",
        "isFlaggedFraud is a weak rule-based flag and should be dropped from modeling because it is target-adjacent and not a causal signal.",
    ]

    report = {
        "dataset_summary": {
            "rows": int(len(report_df)),
            "columns": list(report_df.columns),
            "class_imbalance": round(float(fraud_ratio), 6),
            "fraud_rows": int(report_df["isFraud"].sum()),
            "legitimate_rows": int((1 - report_df["isFraud"]).sum()),
        },
        "fraud_by_type": _safe_table(fraud_by_type),
        "fraud_by_hour": _safe_table(fraud_by_hour),
        "amount_distribution_by_class": _safe_table(amount_distribution),
        "balance_error_summary": {
            "orig_mean": round(float(report_df["errorBalanceOrig"].mean()), 6),
            "orig_std": round(float(report_df["errorBalanceOrig"].std(ddof=1)), 6),
            "dest_mean": round(float(report_df["errorBalanceDest"].mean()), 6),
            "dest_std": round(float(report_df["errorBalanceDest"].std(ddof=1)), 6),
            "orig_zero_after_rate": round(float((report_df["newbalanceOrig"] == 0).mean()), 6),
            "dest_zero_before_rate": round(float((report_df["oldbalanceDest"] == 0).mean()), 6),
        },
        "correlations": corr_records,
        "leakage_risks": leakage_risks,
        "fraud_only_in_types": sorted(report_df.loc[report_df["isFraud"] == 1, "type"].dropna().unique().tolist()),
        "status": "passed",
    }

    return report


def write_eda_outputs(df: pd.DataFrame, output_dir: str | Path = "reports") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_eda_report(df)
    (output_dir / "eda.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fraud_by_type = pd.DataFrame(report["fraud_by_type"])
    fraud_by_hour = pd.DataFrame(report["fraud_by_hour"])
    fraud_by_type.to_csv(output_dir / "fraud_by_type.csv", index=False)
    fraud_by_hour.to_csv(output_dir / "fraud_by_hour.csv", index=False)

    def table_to_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows_"
        headers = list(df.columns)
        rows = [headers] + [list(map(str, row)) for row in df.itertuples(index=False, name=None)]
        widths = [max(len(str(r[i])) for r in rows) for i in range(len(headers))]
        lines = []
        def fmt_row(r):
            return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))
        lines.append(fmt_row(rows[0]))
        lines.append(" | ".join("-" * widths[i] for i in range(len(headers))))
        for r in rows[1:]:
            lines.append(fmt_row(r))
        return "\n".join(lines)

    markdown_lines = [
        "# EDA summary",
        "",
        f"- Fraud rate: {report['dataset_summary']['class_imbalance']:.6%}",
        f"- Total rows: {report['dataset_summary']['rows']}",
        f"- Fraud rows: {report['dataset_summary']['fraud_rows']}",
        f"- Legitimate rows: {report['dataset_summary']['legitimate_rows']}",
        "",
        "## Fraud by transaction type",
        "",
        table_to_markdown(fraud_by_type),
        "",
        "## Fraud by hour",
        "",
        table_to_markdown(fraud_by_hour.head(12)),
        "",
        "## Leakage risk notes",
        "",
        *[f"- {note}" for note in report["leakage_risks"]],
    ]
    (output_dir / "eda.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return report


def run_eda(raw_path: str | Path = RAW_DATA_PATH) -> dict:
    df = pd.read_csv(raw_path)
    return write_eda_outputs(df)


if __name__ == "__main__":
    run_eda()
