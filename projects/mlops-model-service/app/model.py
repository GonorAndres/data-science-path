"""Model loader. Wraps the Predictor from the NLP project."""

import os
from pathlib import Path
from typing import Any


MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "best"),
    )
)


_predictor: Any | None = None


def get_predictor():
    """Lazy-load the predictor on first use."""
    global _predictor
    if _predictor is not None:
        return _predictor
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train the NLP model first or set MODEL_PATH env var."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    labels = ["negative", "neutral", "positive"]

    class _Pred:
        def predict(self, text: str) -> dict:
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**tokens).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            idx = int(torch.argmax(logits).item())
            return {
                "label": labels[idx],
                "confidence": probs[idx],
                "probs": dict(zip(labels, probs)),
            }

    _predictor = _Pred()
    return _predictor


def is_loaded() -> bool:
    return _predictor is not None
