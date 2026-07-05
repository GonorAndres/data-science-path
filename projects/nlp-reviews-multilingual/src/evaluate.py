"""Full evaluation: confusion matrix, per-language metrics, error dump."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import LABELS, load_split
from .predict import Predictor


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "evaluation"


def evaluate() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    test = load_split("test")
    predictor = Predictor()

    preds = [predictor.predict_label_id(t) for t in test.texts]

    overall_f1 = f1_score(test.labels, preds, average="macro")
    print(f"overall macro F1: {overall_f1:.3f}")

    report = classification_report(
        test.labels,
        preds,
        target_names=LABELS,
        digits=3,
    )
    (ARTIFACT_DIR / "classification_report.txt").write_text(report)
    print(report)

    cm = confusion_matrix(test.labels, preds)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        ARTIFACT_DIR / "confusion_matrix.csv"
    )

    per_lang = {}
    for lang in set(test.languages):
        idx = [i for i, lg in enumerate(test.languages) if lg == lang]
        y = [test.labels[i] for i in idx]
        p = [preds[i] for i in idx]
        per_lang[lang] = {
            "n": len(idx),
            "macro_f1": f1_score(y, p, average="macro"),
        }
    (ARTIFACT_DIR / "per_language_metrics.json").write_text(
        json.dumps(per_lang, indent=2)
    )
    print("per-language:", per_lang)

    errors = [
        {
            "text": test.texts[i],
            "true": LABELS[test.labels[i]],
            "pred": LABELS[preds[i]],
            "language": test.languages[i],
        }
        for i, (y, p) in enumerate(zip(test.labels, preds))
        if y != p
    ]
    pd.DataFrame(errors).sample(
        n=min(len(errors), 100), random_state=0
    ).to_csv(ARTIFACT_DIR / "error_samples.csv", index=False)
    print(f"wrote {ARTIFACT_DIR / 'error_samples.csv'}")


if __name__ == "__main__":
    evaluate()
