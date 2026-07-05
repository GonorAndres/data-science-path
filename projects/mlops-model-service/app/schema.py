"""Pydantic request and response models."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probs: dict[str, float]
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
