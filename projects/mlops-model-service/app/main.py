"""FastAPI application for the multilingual review classifier."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from .logging_setup import setup_logging
from .metrics import (
    INPUT_LENGTH,
    PREDICTIONS_TOTAL,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    render_metrics,
)
from .model import MODEL_VERSION, get_predictor, is_loaded
from .schema import (
    BatchPredictRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_predictor()
        logger.info("model loaded at startup")
    except FileNotFoundError as e:
        logger.warning(f"model not yet available: {e}. Health will report degraded.")
    yield


app = FastAPI(
    title="Multilingual Review Classifier",
    description="Sentiment classification (positive / neutral / negative) "
    "for Spanish and English product reviews.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
        duration = time.perf_counter() - start
        REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
        REQUESTS_TOTAL.labels(
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        )
        return response
    except Exception:
        REQUESTS_TOTAL.labels(endpoint=request.url.path, status=500).inc()
        logger.exception("request failed", extra={"request_id": request_id})
        raise


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if is_loaded() else "degraded",
        model_loaded=is_loaded(),
        model_version=MODEL_VERSION if is_loaded() else None,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest):
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    INPUT_LENGTH.observe(len(payload.text))
    result = predictor.predict(payload.text)
    PREDICTIONS_TOTAL.labels(label=result["label"]).inc()

    return PredictResponse(
        label=result["label"],
        confidence=float(result["confidence"]),
        probs={k: float(v) for k, v in result["probs"].items()},
        model_version=MODEL_VERSION,
        request_id=request.state.request_id,
    )


@app.post("/predict/batch")
async def predict_batch(request: Request, payload: BatchPredictRequest):
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    results = []
    for text in payload.texts:
        INPUT_LENGTH.observe(len(text))
        result = predictor.predict(text)
        PREDICTIONS_TOTAL.labels(label=result["label"]).inc()
        results.append(result)

    return {
        "results": results,
        "model_version": MODEL_VERSION,
        "request_id": request.state.request_id,
    }
