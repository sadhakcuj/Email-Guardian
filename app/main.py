"""FastAPI service: spam classification with auth, rate limiting, metrics and structured logs."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import Settings, get_settings
from app.model import SpamModel
from app.schemas import (
    BatchItem,
    BatchRequest,
    BatchResponse,
    FeedbackRequest,
    FeedbackResponse,
    Message,
    PredictResponse,
)

APP_DIR = Path(__file__).resolve().parent
settings = get_settings()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id", "path", "status", "ms", "verdict", "n_items"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


configure_logging(settings.log_level)
log = logging.getLogger("spam.api")

REQUESTS = Counter("spam_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("spam_request_seconds", "Request latency", ["path"])
VERDICTS = Counter("spam_verdicts_total", "Verdicts returned", ["verdict"])

limiter = Limiter(key_func=get_remote_address)
_feedback_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.model = SpamModel.load(s.model_path, s.metrics_path, s.review_floor)
    if not s.api_key_set:
        log.warning("SPAM_API_KEYS is empty: API authentication is DISABLED. Set it in production.")
    log.info("model loaded: %s (threshold %.4f)", app.state.model.name, app.state.model.threshold)
    yield


app = FastAPI(
    title="Spam Classifier API",
    version="2.0.0",
    description="Classifies short messages and emails as spam, ham, or needs-review.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"request_id": request_id, "path": request.url.path})
        response = JSONResponse({"detail": "Internal server error", "request_id": request_id}, status_code=500)
    elapsed = time.perf_counter() - start
    path = request.scope.get("route").path if request.scope.get("route") else "unmatched"
    REQUESTS.labels(path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Message text is deliberately never logged; only metadata.
    log.info(
        "request",
        extra={"request_id": request_id, "path": request.url.path, "status": response.status_code, "ms": round(elapsed * 1000, 1)},
    )
    return response


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header), s: Settings = Depends(get_settings)) -> None:
    allowed = s.api_key_set
    if not allowed:
        return  # auth disabled (dev)
    if not key or not any(secrets.compare_digest(key, k) for k in allowed):
        raise HTTPException(status_code=401, detail="Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})


def get_model(request: Request) -> SpamModel:
    return request.app.state.model


def _text_or_422(msg: Message, max_chars: int) -> str:
    text = msg.combined()
    if not text:
        raise HTTPException(status_code=422, detail="Provide 'text', or 'subject' and/or 'body'.")
    if len(text) > max_chars:
        raise HTTPException(status_code=413, detail=f"Text exceeds {max_chars} characters.")
    return text


# ---- Operational endpoints ----------------------------------------------------------------


@app.get("/healthz", tags=["ops"], summary="Liveness probe")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"], summary="Readiness probe (model loaded)")
def readyz(model: SpamModel = Depends(get_model)) -> dict:
    return {"status": "ready", "model": model.name}


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics(_: None = Depends(require_api_key)) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/model", tags=["model"], summary="Model details and measured performance")
def model_info(model: SpamModel = Depends(get_model)) -> dict:
    m = model.metrics
    return {
        "model": model.name,
        "threshold": model.threshold,
        "review_floor": model.review_floor,
        "trained_at": model.trained_at,
        "dataset": m.get("dataset"),
        "held_out_test": m.get("held_out_test"),
        "caveat": m.get("caveat"),
    }


# ---- Prediction endpoints -----------------------------------------------------------------


def _predict_one(request: Request, msg: Message, model: SpamModel, s: Settings) -> PredictResponse:
    text = _text_or_422(msg, s.max_text_chars)
    pred = model.predict([text])[0]
    VERDICTS.labels(pred.verdict).inc()
    log.info("predict", extra={"request_id": request.state.request_id, "verdict": pred.verdict})
    return PredictResponse(
        score=pred.score,
        verdict=pred.verdict,
        is_spam=pred.verdict == "spam",
        threshold=round(model.threshold, 4),
        signals=pred.signals,
        model=model.name,
        request_id=request.state.request_id,
    )


@app.post("/v1/predict", response_model=PredictResponse, tags=["predict"], dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_api)
def predict(request: Request, msg: Message, model: SpamModel = Depends(get_model), s: Settings = Depends(get_settings)):
    """Classify one message. Used by automation (n8n, Zapier, scripts)."""
    return _predict_one(request, msg, model, s)


@app.post("/v1/predict/batch", response_model=BatchResponse, tags=["predict"], dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_api)
def predict_batch(request: Request, body: BatchRequest, model: SpamModel = Depends(get_model), s: Settings = Depends(get_settings)):
    """Classify up to ``SPAM_MAX_BATCH_SIZE`` messages in one call."""
    if len(body.items) > s.max_batch_size:
        raise HTTPException(status_code=413, detail=f"Batch exceeds {s.max_batch_size} items.")
    texts = [_text_or_422(item, s.max_text_chars) for item in body.items]
    preds = model.predict(texts)
    for p in preds:
        VERDICTS.labels(p.verdict).inc()
    log.info("predict_batch", extra={"request_id": request.state.request_id, "n_items": len(texts)})
    return BatchResponse(
        results=[BatchItem(score=p.score, verdict=p.verdict, is_spam=p.verdict == "spam", signals=p.signals) for p in preds],
        threshold=round(model.threshold, 4),
        model=model.name,
        request_id=request.state.request_id,
    )


@app.post("/demo/predict", response_model=PredictResponse, tags=["demo"])
@limiter.limit(lambda: get_settings().rate_limit_demo)
def demo_predict(request: Request, msg: Message, model: SpamModel = Depends(get_model), s: Settings = Depends(get_settings)):
    """Unauthenticated endpoint backing the web UI. Tighter rate limit; disabled when SPAM_DEMO_ENABLED=false."""
    if not s.demo_enabled:
        raise HTTPException(status_code=404, detail="Demo endpoint disabled.")
    return _predict_one(request, msg, model, s)


@app.post("/v1/feedback", response_model=FeedbackResponse, tags=["feedback"], dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_api)
def feedback(request: Request, body: FeedbackRequest, s: Settings = Depends(get_settings)):
    """Record a human correction. Stored as JSONL for the next retraining run."""
    text = _text_or_422(body, s.max_text_chars)
    record = {
        "ts": time.time(),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "label": 1 if body.correct_label == "spam" else 0,
        "request_id": body.request_id,
    }
    s.feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with _feedback_lock:
        existing = 0
        seen: set[str] = set()
        if s.feedback_path.exists():
            for line in s.feedback_path.read_text().splitlines():
                existing += 1
                try:
                    seen.add(json.loads(line)["sha256"])
                except (ValueError, KeyError):
                    continue
        if record["sha256"] in seen:
            return FeedbackResponse(stored=False, total_feedback=existing)
        with s.feedback_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    return FeedbackResponse(stored=True, total_feedback=existing + 1)


# ---- UI -----------------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html", media_type="text/html")
