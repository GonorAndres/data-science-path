"""Pydantic schema validation tests."""

import pytest
from pydantic import ValidationError

from app.schema import BatchPredictRequest, PredictRequest


def test_predict_request_valid():
    r = PredictRequest(text="Me encanto")
    assert r.text == "Me encanto"


def test_predict_request_empty_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_batch_request_too_large_rejected():
    with pytest.raises(ValidationError):
        BatchPredictRequest(texts=["a"] * 101)


def test_batch_request_empty_rejected():
    with pytest.raises(ValidationError):
        BatchPredictRequest(texts=[])
