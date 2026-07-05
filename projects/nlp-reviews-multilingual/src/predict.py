"""Single-input inference wrapper for the demo and MLOps service."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABELS


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "transformer" / "best"


class Predictor:
    def __init__(self, model_dir: str | Path = MODEL_DIR):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"{model_dir} not found. Train the model first (python -m src.train)."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        if not text or not isinstance(text, str):
            return {"label": "neutral", "confidence": 0.0, "probs": {l: 1/3 for l in LABELS}}
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**tokens).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        label_id = int(torch.argmax(logits).item())
        return {
            "label": LABELS[label_id],
            "confidence": probs[label_id],
            "probs": dict(zip(LABELS, probs)),
        }

    def predict_label_id(self, text: str) -> int:
        return LABELS.index(self.predict(text)["label"])


if __name__ == "__main__":
    p = Predictor()
    examples = [
        "El producto llego rapido y funciona perfecto. Super recomendado!",
        "No me gusto, la calidad es pesima y tardo en llegar.",
        "Ni bueno ni malo, cumple su funcion basica.",
        "The phone is great, but the charger died after a week.",
    ]
    for text in examples:
        out = p.predict(text)
        print(f"{text[:50]:50s} -> {out['label']} ({out['confidence']:.3f})")
