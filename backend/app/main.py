import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes.monitoring import router as monitoring_router
from backend.app.api.routes.prediction import router as prediction_router
from backend.app.api.routes.analytics import router as analytics_router
from backend.app.api.routes.model import router as model_router
from backend.app.config.settings import API_PREFIX, APP_NAME, APP_VERSION, FRONTEND_ORIGINS
from backend.app.services.monitoring_service import runtime_metrics

class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            event["error_type"] = record.exc_info[0].__name__
        return json.dumps(event)


handler = logging.StreamHandler()
handler.setFormatter(JsonLogFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Score PaySim transaction inputs with the saved fraud detection pipeline.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(FRONTEND_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    runtime_metrics.request(elapsed_ms, response.status_code)
    logger.info(
        "http_request method=%s path=%s status=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_api_error path=%s error_type=%s",
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    if request.method == "POST" and request.url.path == f"{API_PREFIX}/predict":
        runtime_metrics.prediction("prediction_failures")
    return await request_validation_exception_handler(request, exc)


@app.get("/health", tags=["health"])
def health() -> dict:
    from backend.app.ml.model_loader import load_model_artifacts, load_model_metadata

    try:
        load_model_artifacts()
        metadata = load_model_metadata()
        model_loaded = True
        version = metadata.get("version", "unknown")
    except (FileNotFoundError, ValueError, TypeError):
        model_loaded = False
        version = "unavailable"
    return {
        "status": "ok" if model_loaded else "degraded",
        "service": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": model_loaded,
        "model_version": version,
    }


app.include_router(prediction_router, prefix=API_PREFIX)
app.include_router(monitoring_router, prefix=API_PREFIX)
app.include_router(model_router, prefix=API_PREFIX)
app.include_router(analytics_router, prefix=API_PREFIX)
