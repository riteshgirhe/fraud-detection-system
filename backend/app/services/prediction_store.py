from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from backend.app.config.settings import PROJECT_ROOT

PREDICTIONS_PATH = Path(
    os.getenv(
        "FRAUD_PREDICTIONS_PATH",
        str(PROJECT_ROOT / "data" / "predictions" / "predictions.jsonl"),
    )
)
_lock = RLock()
_records: dict[str, dict[str, Any]] | None = None


def _load_records() -> dict[str, dict[str, Any]]:
    global _records
    if _records is not None:
        return _records
    records: dict[str, dict[str, Any]] = {}
    if PREDICTIONS_PATH.is_file():
        with PREDICTIONS_PATH.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid prediction log JSON at line {line_number}."
                    ) from exc
                record_id = record.get("transaction_id")
                if not isinstance(record_id, str):
                    raise ValueError(
                        f"Prediction log entry at line {line_number} has no transaction_id."
                    )
                records[record_id] = record
    _records = records
    return records


def append_prediction(record: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        records = _load_records()
        if record["transaction_id"] in records:
            raise ValueError("Prediction transaction_id already exists.")
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        records[record["transaction_id"]] = record
        return dict(record)


def list_predictions(
    *,
    offset: int = 0,
    limit: int = 50,
    is_fraud: bool | None = None,
    labeled: bool | None = None,
) -> dict[str, Any]:
    with _lock:
        values = list(_load_records().values())
        if is_fraud is not None:
            values = [row for row in values if row["is_fraud"] is is_fraud]
        if labeled is not None:
            values = [row for row in values if (row.get("true_label") is not None) is labeled]
        values.sort(key=lambda row: row["timestamp"], reverse=True)
        return {
            "items": values[offset : offset + limit],
            "total": len(values),
            "offset": offset,
            "limit": limit,
        }


def get_all_predictions() -> list[dict[str, Any]]:
    with _lock:
        return list(_load_records().values())


def attach_label(transaction_id: str, label: int) -> dict[str, Any]:
    with _lock:
        records = _load_records()
        record = records.get(transaction_id)
        if record is None:
            raise KeyError(transaction_id)
        updated = dict(record)
        updated["true_label"] = label
        updated["labeled_at"] = datetime.now(timezone.utc).isoformat()
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=PREDICTIONS_PATH.parent,
            prefix="predictions-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in records.values():
                selected = updated if row["transaction_id"] == transaction_id else row
                handle.write(json.dumps(selected, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, PREDICTIONS_PATH)
        records[transaction_id] = updated
        return dict(updated)
