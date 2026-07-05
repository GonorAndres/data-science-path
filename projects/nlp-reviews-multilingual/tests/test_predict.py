"""Stress tests for the Predictor interface. Skipped if the model is not yet trained."""

from pathlib import Path

import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "transformer" / "best"
pytestmark = pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason="Transformer model not trained yet. Run `python -m src.train` first.",
)


@pytest.fixture(scope="module")
def predictor():
    from src.predict import Predictor
    return Predictor()


def test_predict_returns_schema(predictor):
    out = predictor.predict("Me encanto el producto!")
    assert set(out.keys()) == {"label", "confidence", "probs"}
    assert out["label"] in {"negative", "neutral", "positive"}
    assert 0 <= out["confidence"] <= 1
    assert abs(sum(out["probs"].values()) - 1) < 1e-3


def test_predict_handles_empty_string(predictor):
    out = predictor.predict("")
    assert out["label"] in {"negative", "neutral", "positive"}


def test_predict_handles_very_long_text(predictor):
    text = "bueno " * 1000
    out = predictor.predict(text)
    assert out["label"] in {"negative", "neutral", "positive"}


def test_predict_handles_emoji(predictor):
    out = predictor.predict("10/10")
    assert out["label"] in {"negative", "neutral", "positive"}


def test_predict_negation_spanish(predictor):
    # "No esta mal" is a mild positive; acceptable outputs: positive, neutral.
    out = predictor.predict("No esta mal el producto, me gusto.")
    assert out["label"] in {"positive", "neutral"}


def test_predict_negation_english(predictor):
    out = predictor.predict("Not bad at all, I am satisfied with it.")
    assert out["label"] in {"positive", "neutral"}
