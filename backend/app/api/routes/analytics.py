import json

from fastapi import APIRouter, HTTPException

from backend.app.config.settings import PROJECT_ROOT

router = APIRouter(tags=["reports"])


def _read_report(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Report not found: {relative_path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@router.get("/reports/eda")
def eda_report() -> dict:
    return _read_report("reports/eda.json")


@router.get("/reports/comparison")
def comparison_report() -> dict:
    return _read_report("reports/model_comparison.json")