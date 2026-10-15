from fastapi import APIRouter

from backend.app.services.monitoring_service import (
    drift_report,
    monitoring_summary,
    runtime_metrics,
)

router = APIRouter(tags=["monitoring"])


@router.get("/monitoring/metrics")
def metrics() -> dict:
    return runtime_metrics.snapshot()


@router.get("/monitoring/summary")
def summary() -> dict:
    return monitoring_summary()


@router.get("/monitoring/drift")
def drift() -> dict:
    return drift_report()
