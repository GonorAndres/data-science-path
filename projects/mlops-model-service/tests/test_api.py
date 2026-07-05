"""API-level smoke tests. Mocks the predictor so we do not depend on weights."""

import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_predictor():
    class FakePredictor:
        def predict(self, _text: str) -> dict:
            return {
                "label": "positive",
                "confidence": 0.9,
                "probs": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
            }
    return FakePredictor()


@pytest.fixture
def client(mock_predictor):
    with patch("app.model.get_predictor", return_value=mock_predictor), \
         patch("app.model.is_loaded", return_value=True):
        # Ensure app imports after patches for lifespan to pick up mock
        sys.modules.pop("app.main", None)
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_endpoint(client):
    r = client.post("/predict", json={"text": "Me encanto el producto"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "positive"
    assert 0 <= body["confidence"] <= 1
    assert set(body["probs"].keys()) == {"negative", "neutral", "positive"}
    assert "request_id" in body


def test_predict_rejects_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_rejects_too_long_text(client):
    r = client.post("/predict", json={"text": "x" * 11_000})
    assert r.status_code == 422


def test_batch_endpoint(client):
    r = client.post(
        "/predict/batch",
        json={"texts": ["El producto es excelente", "No me gusto nada"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


def test_request_id_is_echoed(client):
    rid = "test-request-id-123"
    r = client.post(
        "/predict",
        json={"text": "hello"},
        headers={"X-Request-ID": rid},
    )
    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == rid
